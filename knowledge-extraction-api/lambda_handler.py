import json
import base64
import os
import tempfile
import yaml
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import boto3
from botocore.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AWS clients - lazy initialization (only when needed for Lambda)
_s3_client = None
SKILLS_BUCKET = os.environ.get('SKILLS_BUCKET')


def get_s3_client():
    """Get S3 client (lazy initialization for local execution support)"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client

# Model configuration
BEDROCK_MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
MAX_TOKENS = 200000
DEFAULT_MAX_OUTPUT_TOKENS = 16000
MAX_INPUT_CHARS = 300000

# CORS headers for all responses
CORS_HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token'
}

# ============================================================================
# DOCUMENT PARSER - Multi-format support
# ============================================================================

# Check for optional dependencies
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def parse_document(file_path: str, filename: str) -> str:
    """Parse document to plain text based on file extension.

    Supports: .txt, .md, .markdown, .rst, .json, .xml, .html, .csv, .pdf, .docx
    """
    ext = Path(filename).suffix.lower()

    # Text-based formats
    if ext in ['.txt', '.md', '.markdown', '.rst', '.json', '.xml', '.html', '.htm', '.csv']:
        return parse_text(file_path)
    # PDF
    elif ext == '.pdf':
        return parse_pdf(file_path)
    # Word documents
    elif ext == '.docx':
        return parse_docx(file_path)
    else:
        # Try to detect format from file content
        return parse_unknown(file_path, ext)


def parse_text(file_path: str) -> str:
    """Parse plain text files with encoding fallback"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode text file with any supported encoding")


def parse_pdf(file_path: str) -> str:
    """Parse PDF files using PyPDF2"""
    if not HAS_PYPDF2:
        raise ImportError("PDF support requires PyPDF2. Install with: pip install PyPDF2")

    text_parts = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        logger.info(f"   📄 PDF has {num_pages} pages")

        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")

    content = '\n\n'.join(text_parts)

    if not content.strip():
        raise ValueError("PDF appears to be empty or contains only images (no extractable text)")

    return content


def parse_docx(file_path: str) -> str:
    """Parse DOCX files using python-docx"""
    if not HAS_DOCX:
        raise ImportError("DOCX support requires python-docx. Install with: pip install python-docx")

    doc = DocxDocument(file_path)
    text_parts = []

    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)

    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_parts.append(row_text)

    content = '\n\n'.join(text_parts)

    if not content.strip():
        raise ValueError("DOCX appears to be empty")

    return content


def parse_unknown(file_path: str, ext: str) -> str:
    """Attempt to parse unknown file format"""
    # Try as text first
    try:
        return parse_text(file_path)
    except:
        pass

    # Check magic bytes
    with open(file_path, 'rb') as f:
        header = f.read(8)

    if header.startswith(b'%PDF'):
        return parse_pdf(file_path)
    elif header.startswith(b'PK'):  # ZIP-based (docx, xlsx, etc.)
        try:
            return parse_docx(file_path)
        except:
            pass

    raise ValueError(f"Unsupported file format: {ext}\nSupported: .txt, .md, .pdf, .docx")


# ============================================================================
# MULTIPART FORM DATA PARSER
# ============================================================================

def parse_multipart_form_data(body: str, content_type: str) -> Dict[str, Any]:
    """
    Parse multipart/form-data from API Gateway
    
    Returns:
        {
            'file': bytes,
            'filename': str,
            'enterprise_name': str (optional)
        }
    """
    # Extract boundary from content-type header
    boundary = None
    for part in content_type.split(';'):
        if 'boundary=' in part:
            boundary = part.split('boundary=')[1].strip()
            break
    
    if not boundary:
        raise ValueError("No boundary found in Content-Type header")
    
    # Decode body if base64 encoded (API Gateway does this)
    if body:
        try:
            body_bytes = base64.b64decode(body)
        except:
            body_bytes = body.encode('utf-8')
    else:
        raise ValueError("Empty body")
    
    # Parse multipart parts
    parts = body_bytes.split(f'--{boundary}'.encode())
    
    result = {}
    
    for part in parts:
        if not part or part == b'--\r\n' or part == b'--':
            continue
        
        # Split headers and content
        if b'\r\n\r\n' in part:
            headers_section, content = part.split(b'\r\n\r\n', 1)
        else:
            continue
        
        headers = headers_section.decode('utf-8', errors='ignore')
        
        # Extract field name and filename
        if 'Content-Disposition' in headers:
            disposition_line = [line for line in headers.split('\r\n') 
                               if 'Content-Disposition' in line][0]
            
            # Extract name
            if 'name="' in disposition_line:
                name_start = disposition_line.index('name="') + 6
                name_end = disposition_line.index('"', name_start)
                field_name = disposition_line[name_start:name_end]
            else:
                continue
            
            # Extract filename if present
            filename = None
            if 'filename="' in disposition_line:
                filename_start = disposition_line.index('filename="') + 10
                filename_end = disposition_line.index('"', filename_start)
                filename = disposition_line[filename_start:filename_end]
            
            # Clean up content (remove trailing boundary markers)
            content = content.rstrip(b'\r\n')
            
            if field_name == 'file':
                result['file'] = content
                if filename:
                    result['filename'] = filename
            elif field_name == 'enterprise_name':
                result['enterprise_name'] = content.decode('utf-8').strip()
    
    return result


# ============================================================================
# CANONICAL TEMPLATES - Production-Ready Format
# ============================================================================

METADATA_JSON_TEMPLATE = """{
  "domain": "<inferred enterprise domain>",
  "source_documents": [
    {
      "document_name": "<source file name>",
      "document_type": "<SOP | Policy | Manual | Email | Regulation | Other>",
      "document_version": "<if available, else null>",
      "document_owner": "<team or function if inferable, else null>"
    }
  ],
  "extracted_intents": ["<complete intent statement>"],
  "confidence_level": "<high | medium | low>",
  "assumptions": ["<explicit assumption>"],
  "limitations": ["<what NOT covered>"],
  "extraction_timestamp": "<ISO-8601 UTC>",
  "extraction_version": "knowledge-capture-v1"
}"""

RULES_YAML_TEMPLATE = """rules:
  - rule_name: "descriptive_name"
    rule: "Clear statement of requirement"
    authority: "Who can override (e.g., Manager only)"
    consequence: "What happens if violated"
    sop_reference: "Section X.X: [exact quote]"
    violation_severity: "<critical | high | medium | low>"
    signal_words: "<MUST | NEVER | ALWAYS | REQUIRED | IF/THEN>"
"""

DEFINITIONS_YAML_TEMPLATE = """definitions:
  - term: "<domain-specific term>"
    definition: "<clear meaning>"
    synonyms: ["<alternate terms>"]
    usage_context: "<when this term applies>"
"""

EXAMPLES_YAML_TEMPLATE = """examples:
  - scenario: "Descriptive scenario name"
    customer_input: "Exact customer statement"
    agent_workflow:
      - step: "Parse request"
        result: {key: "value"}
      - step: "Execute action"
        action: "What agent does"
        result: "Outcome"
    skills_chained: ["skill-1", "skill-2"]
    expected_outcome: "Final result"
    test_data_used: ["DATA-001", "PROD-123"]
"""


@dataclass
class KnowledgeType:
    """Knowledge types for extraction"""
    PROCEDURE = "procedure"
    DECISION_RULE = "decision_rule"
    DEFINITION = "definition"
    EXAMPLE = "example"
    WORKFLOW = "workflow"
    CONSTRAINT = "constraint"


# Script template for generating executable Python skills
SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
{skill_name} - {skill_description}
Generated from: {knowledge_source}
Version: {version}
"""

from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {class_docstring}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.logger = logger
        self._initialize()

    def _initialize(self):
        """Initialize skill-specific resources."""
{initialization_code}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill.

        Args:
            input_data: {input_description}

        Returns:
            {return_description}
        """
        try:
            # Validate input
            if not self._validate_input(input_data):
                return {{"error": "Invalid input data", "success": False}}

            # Main logic
{main_execution_logic}

            return result

        except Exception as e:
            self.logger.error(f"Execution error: {{e}}")
            return {{"error": str(e), "success": False}}

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure."""
{validation_logic}
        return True
{helper_methods}

    def get_metadata(self) -> Dict[str, Any]:
        """Return skill metadata."""
        return {{
            "name": "{skill_name}",
            "version": "{version}",
            "description": "{skill_description}",
            "required_inputs": {required_inputs},
            "optional_inputs": {optional_inputs},
            "outputs": {outputs}
        }}


# Example usage
if __name__ == "__main__":
    import json

    # Initialize the skill
    skill = {class_name}()

    # Example input
    example_input = {{
        # Add your test input here based on required_inputs
    }}

    # Execute
    result = skill.execute(example_input)
    print(json.dumps(result, indent=2))
'''


@dataclass
class ExtractedKnowledge:
    """Container for extracted knowledge"""
    type: str
    content: str
    source_document: str
    source_section: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    requires_review: bool = False
    source_enterprise: Optional[str] = None
    skill_boundary: Optional[str] = None
    user_intention: Optional[str] = None  
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentContext:
    """Document being processed"""
    filename: str
    file_type: str
    content: str
    enterprise: Optional[str] = None


@dataclass
class SkillMetadata:
    """Skill metadata"""
    name: str
    description: str
    version: str
    source_documents: List[str]
    confidence: float
    created_date: str
    enterprise: Optional[str] = None
    domain: Optional[str] = None
    knowledge_extraction_stats: Dict[str, int] = field(default_factory=dict)


class BedrockLLMClient:
    """AWS Bedrock client"""
    
    def __init__(self, model_id: str = BEDROCK_MODEL, region: str = None):
        self.model_id = model_id
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        logger.info(f"🔧 Initializing Bedrock client")

        config = Config(
            connect_timeout=30,
            read_timeout=900,
            retries={"max_attempts": 5, "mode": "adaptive"}
        )
        
        try:
            self.client = boto3.client('bedrock-runtime', region_name=self.region, config=config)
            logger.info("✅ Bedrock client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}", exc_info=True)
            raise
    
    def invoke(self, prompt: str, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        """Invoke model"""
        prompt_length = len(prompt)
        
        if prompt_length > MAX_INPUT_CHARS:
            logger.warning(f"⚠️  Truncating prompt from {prompt_length} chars")
            prompt = prompt[:MAX_INPUT_CHARS] + "\n\n[TRUNCATED]"
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            content = "".join(
                block.get("text", "") 
                for block in response_body.get('content', []) 
                if block.get("type") == "text"
            )
            
            return content
        except Exception as e:
            logger.error(f"❌ Invocation failed: {e}", exc_info=True)
            raise RuntimeError(f"Bedrock invocation failed: {e}")


class KnowledgeCaptureOrchestrator:
    """
    PRODUCTION-READY SKILL GENERATOR
    
    Generates agent-executable skills that meet:
    - Natural language triggers
    - Comprehensive rule extraction
    - Realistic examples with workflows
    - Data dependencies
    - Skill composition
    - Error handling
    """
    
    def __init__(self, model_id: str = BEDROCK_MODEL, region: str = None):
        logger.info("="*70)
        logger.info("🚀 PRODUCTION-READY SKILL GENERATOR")
        logger.info("="*70)
        self.client = BedrockLLMClient(model_id=model_id, region=region)
        self.model = model_id
        logger.info(f"✅ Ready (Model: {self.model})")
        logger.info("")
        logger.info("QUALITY STANDARDS:")
        logger.info("  • Natural language triggers (customer speech patterns)")
        logger.info("  • Comprehensive rules (MUST/NEVER/ALWAYS/REQUIRED)")
        logger.info("  • Realistic examples (full workflows)")
        logger.info("  • Data dependencies specified")
        logger.info("  • Skill composition mapped")
        logger.info("  • Error handling included")
        logger.info("")
        
    def process_documents(
        self, 
        document_paths: List[str],
        enterprise_name: Optional[str] = None,
        output_dir: str = "./extracted_knowledge"
    ) -> Dict[str, Any]:
        """Main entry point"""
        
        logger.info("="*70)
        logger.info("📚 PROCESSING STARTED")
        logger.info("="*70)
        logger.info(f"Documents: {len(document_paths)}")
        logger.info(f"Enterprise: {enterprise_name or 'Will be inferred'}")
        logger.info(f"Output: {output_dir}")
        logger.info("")
        
        try:
            # PHASE 1: Upload
            logger.info("📤 PHASE 1: UPLOAD")
            logger.info("-"*70)
            documents = self._upload_documents(document_paths, enterprise_name)
            logger.info(f"✅ Uploaded {len(documents)} document(s)")
            logger.info("")
            
            # PHASE 2: Extract
            logger.info("⚡ PHASE 2: EXTRACT")
            logger.info("-"*70)
            extracted_knowledge = self._extract_knowledge(documents)
            logger.info(f"✅ Extracted {len(extracted_knowledge)} items")
            logger.info("")
            
            # PHASE 3: Review
            logger.info("📋 PHASE 3: REVIEW")
            logger.info("-"*70)
            consolidated = self._review_knowledge(extracted_knowledge)
            logger.info(f"✅ Reviewed {len(consolidated)} unique items")
            logger.info("")
            
            # PHASE 4: Generate
            logger.info("⚙️  PHASE 4: GENERATE")
            logger.info("-"*70)
            skills = self._generate_skills(consolidated, enterprise_name)
            logger.info(f"✅ Generated {len(skills)} skill(s)")
            logger.info("")
            
            # PHASE 5: Complete
            logger.info("✅ PHASE 5: COMPLETE")
            logger.info("-"*70)
            self._complete_outputs(skills, output_dir)
            logger.info(f"✅ Saved to {output_dir}")
            logger.info("")
            
            result = {
                "status": "success",
                "documents_processed": len(documents),
                "knowledge_items_extracted": len(consolidated),
                "skills_generated": len(skills),
                "output_directory": output_dir,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("="*70)
            logger.info("✅ COMPLETE")
            logger.info("="*70)
            
            return result
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ FAILED")
            logger.error("="*70)
            logger.error(f"Error: {e}", exc_info=True)
            raise
    
    def _upload_documents(self, document_paths: List[str], enterprise_name: Optional[str]) -> List[DocumentContext]:
        """PHASE 1: Upload - Supports multiple file formats (txt, md, pdf, docx)"""
        documents = []

        for i, path in enumerate(document_paths, 1):
            try:
                filename = os.path.basename(path)
                file_ext = Path(path).suffix.lower()
                logger.info(f"📄 [{i}/{len(document_paths)}] {filename} ({file_ext})")

                # Use the multi-format document parser
                content = parse_document(path, filename)

                doc = DocumentContext(
                    filename=filename,
                    file_type=file_ext,
                    content=content,
                    enterprise=enterprise_name
                )
                documents.append(doc)
                logger.info(f"   ✅ Uploaded ({len(content)} chars extracted)")
                # Show preview of content
                preview = content[:200].replace('\n', ' ')
                logger.info(f"   📄 Preview: {preview}...")

            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")

        return documents
    
    def _extract_knowledge(self, documents: List[DocumentContext]) -> List[ExtractedKnowledge]:
        """PHASE 2: Extract"""
        all_knowledge = []
        
        for i, doc in enumerate(documents, 1):
            try:
                logger.info(f"⚡ [{i}/{len(documents)}] Extracting: {doc.filename}")
                
                knowledge_items = self._extract_from_document(doc)
                
                stats = {}
                for item in knowledge_items:
                    stats[item.type] = stats.get(item.type, 0) + 1
                
                logger.info(f"   Extracted {len(knowledge_items)} items:")
                for k_type, count in sorted(stats.items()):
                    logger.info(f"      • {k_type}: {count}")
                
                all_knowledge.extend(knowledge_items)
                
            except Exception as e:
                logger.error(f"   ❌ Extraction failed: {e}")
        
        return all_knowledge
    
    def _extract_from_document(self, doc: DocumentContext) -> List[ExtractedKnowledge]:
        """Extract with comprehensive rule extraction"""
        
        prompt = f"""Extract knowledge for production-ready skill generation.

DOCUMENT: {doc.filename}
CONTENT: {doc.content[:12000]}

METADATA FORMAT (Use this exact structure):
{METADATA_JSON_TEMPLATE}

COMPREHENSIVE RULE EXTRACTION (CRITICAL):
==========================================
Extract EVERY constraint using these signal words:
- MUST / REQUIRED → Mandatory rule
- NEVER / DO NOT → Prohibition rule  
- ALWAYS → Consistent behavior rule
- IF...THEN → Conditional rule
- ONLY → Restriction rule
- SHOULD / RECOMMENDED → Advisory rule
- BEFORE / AFTER / IMMEDIATELY → Sequencing rule

Example from SOP:
"Walk them to the item (don't just point)"
Becomes:
{{
  "type": "decision_rule",
  "content": "MUST physically walk customer to product location. NEVER just point or give verbal directions only.",
  "section": "Section 1.3",
  "sop_quote": "Walk them to the item (don't just point)"
}}

OUTPUT JSON:
{{
  "inferred_domain": "Domain name",
  "user_intentions": [
    {{
      "intention_id": "hyphenated-id",
      "intention_description": "What user wants",
      "outcome": "Expected result"
    }}
  ],
  "extracted_knowledge": [
    {{
      "type": "decision_rule",
      "content": "Complete rule with signal word",
      "section": "Source section",
      "confidence": 0.9,
      "user_intention": "intention-id",
      "metadata": {{
        "signal_word": "MUST",
        "sop_quote": "Exact quote from document"
      }}
    }}
  ]
}}

Extract 15-25 items. Be COMPREHENSIVE on rules."""

        try:
            logger.info(f"   📝 Sending extraction request ({len(doc.content)} chars)...")
            response = self.client.invoke(prompt, max_tokens=16000)
            logger.info(f"   📥 Received response ({len(response)} chars)")

            response_clean = self._clean_json_response(response)
            logger.info(f"   🔧 Cleaned response, parsing JSON...")

            try:
                result = json.loads(response_clean)
            except json.JSONDecodeError as je:
                logger.error(f"   ❌ JSON parse error: {je}")
                logger.error(f"   Response preview: {response_clean[:500]}...")
                return []

            inferred_domain = result.get("inferred_domain", "Enterprise Operations")
            logger.info(f"   🏢 Domain: {inferred_domain}")
            doc.enterprise = doc.enterprise or inferred_domain

            extracted_items = result.get("extracted_knowledge", [])
            logger.info(f"   📊 Found {len(extracted_items)} items in response")

            knowledge_items = []
            for item in extracted_items:
                knowledge = ExtractedKnowledge(
                    type=item.get("type"),
                    content=item.get("content"),
                    source_document=doc.filename,
                    source_section=item.get("section"),
                    confidence=item.get("confidence", 0.8),
                    metadata=item.get("metadata", {}),
                    source_enterprise=doc.enterprise,
                    skill_boundary=item.get("user_intention"),
                    user_intention=item.get("user_intention")
                )
                knowledge_items.append(knowledge)

            if not any(k.user_intention for k in knowledge_items):
                default_intention = Path(doc.filename).stem.replace("_", "-").lower()
                for k in knowledge_items:
                    k.user_intention = default_intention

            return knowledge_items

        except Exception as e:
            logger.error(f"   ❌ Extraction failed: {e}", exc_info=True)
            return []
    
    def _review_knowledge(self, knowledge: List[ExtractedKnowledge]) -> List[ExtractedKnowledge]:
        """PHASE 3: Review"""
        
        if not knowledge:
            return []
        
        seen = set()
        unique = []
        for item in knowledge:
            key = f"{item.type}:{item.user_intention}:{item.content[:100]}"
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        logger.info(f"   Deduplicated: {len(knowledge)} → {len(unique)}")
        return unique
    
    def _generate_skills(
        self, 
        knowledge: List[ExtractedKnowledge], 
        enterprise_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """PHASE 4: Generate production-ready skills"""
        
        if not knowledge:
            return self._create_fallback_skill(enterprise_name)
        
        # Group by intention
        skills_by_intention = {}
        for item in knowledge:
            intention = item.user_intention or "default"
            if intention not in skills_by_intention:
                skills_by_intention[intention] = []
            skills_by_intention[intention].append(item)
        
        logger.info(f"   Grouped into {len(skills_by_intention)} skill(s)")
        
        generated_skills = []
        for i, (intention, skill_knowledge) in enumerate(skills_by_intention.items(), 1):
            try:
                logger.info(f"⚙️  [{i}/{len(skills_by_intention)}] Generating: {intention}")
                
                skill_output = self._generate_production_skill(intention, skill_knowledge, enterprise_name)
                
                if skill_output:
                    generated_skills.append(skill_output)
                    logger.info(f"   ✅ Generated")
                    
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
        
        if not generated_skills:
            generated_skills = self._create_fallback_skill(enterprise_name, knowledge)
        
        return generated_skills
    
    def _generate_production_skill(
        self,
        intention: str,
        knowledge: List[ExtractedKnowledge],
        enterprise_name: Optional[str]
    ) -> Dict[str, Any]:
        """Generate production-ready skill with all quality requirements"""
        
        procedures = [k for k in knowledge if k.type == KnowledgeType.PROCEDURE]
        decision_rules = [k for k in knowledge if k.type == KnowledgeType.DECISION_RULE]
        definitions = [k for k in knowledge if k.type == KnowledgeType.DEFINITION]
        
        inferred_domain = knowledge[0].source_enterprise if knowledge else enterprise_name or "Enterprise Operations"
        
        # Generate with production quality
        skill_md = self._generate_production_skill_md(intention, procedures, decision_rules, inferred_domain, knowledge)
        rules_yaml = self._generate_production_rules_yaml(decision_rules, knowledge)
        definitions_yaml = self._generate_definitions_yaml(definitions)
        examples_yaml = self._generate_production_examples_yaml(knowledge, intention, procedures)

        # Generate configuration file for customization
        config_yaml = self._generate_config_yaml(intention, decision_rules, inferred_domain, enterprise_name)

        # Generate Python script
        script_py = self._generate_script_for_skill(intention, knowledge, inferred_domain)

        metadata = SkillMetadata(
            name=intention,
            description=f"Production skill: {intention}",
            version="1.0.0",
            source_documents=list(set(k.source_document for k in knowledge)),
            confidence=sum(k.confidence for k in knowledge) / len(knowledge) if knowledge else 0.5,
            created_date=datetime.now().isoformat(),
            enterprise=enterprise_name,
            domain=inferred_domain,
            knowledge_extraction_stats={}
        )

        return {
            "metadata": asdict(metadata),
            "skill_md": skill_md,
            "rules_yaml": rules_yaml,
            "definitions_yaml": definitions_yaml,
            "examples_yaml": examples_yaml,
            "config_yaml": config_yaml,
            "script_py": script_py
        }
    
    def _generate_production_skill_md(
        self,
        intention: str,
        procedures: List[ExtractedKnowledge],
        decision_rules: List[ExtractedKnowledge],
        domain: str,
        all_knowledge: List[ExtractedKnowledge]
    ) -> str:
        """Generate SKILL.md with production quality"""
        
        # Generate NATURAL LANGUAGE triggers
        triggers = self._generate_natural_triggers(intention, all_knowledge)
        description = f"Production skill: {intention.replace('-', ' ')}"
        
        frontmatter = f"""---
name: {intention}
description: {description}
version: 1.0.0
triggers:
"""
        for trigger in triggers[:7]:
            frontmatter += f'  - pattern: "{trigger}"\n'
        frontmatter += "---\n"
        
        md = frontmatter
        
        md += "\n## Overview\n\n"
        md += f"Enables {intention.replace('-', ' ')} in {domain}.\n\n"
        
        md += "## When to Use\n\n"
        md += "Activate when customer expresses:\n"
        for trigger in triggers[:3]:
            md += f'- {trigger}\n'
        md += "\n"
        
        md += "## Instructions\n\n"
        if procedures:
            md += self._extract_steps(procedures[0].content)
        else:
            md += "1. Verify intent\n2. Execute workflow\n3. Confirm completion\n"
        md += "\n"
        
        md += "## Rules\n\n"
        if decision_rules:
            for rule in decision_rules:
                # Extract signal word
                signal = "MUST" if "must" in rule.content.lower() else \
                        "NEVER" if "never" in rule.content.lower() else \
                        "ALWAYS" if "always" in rule.content.lower() else \
                        "REQUIRED" if "required" in rule.content.lower() else "RULE"
                md += f"- **{signal}**: {rule.content}\n"
                if rule.metadata and rule.metadata.get("sop_quote"):
                    md += f"  - Source: {rule.source_section}: \"{rule.metadata['sop_quote']}\"\n"
        else:
            md += "- No explicit constraints documented\n"
        md += "\n"
        
        md += "## Required Data\n\n"
        md += "```yaml\n"
        md += "required_data:\n"
        md += "  - name: primary_data_source\n"
        md += "    type: database\n"
        md += "    fields: [id, name, status]\n"
        md += "```\n\n"
        
        md += "## Skill Composition\n\n"
        md += "```yaml\n"
        md += "skill_composition:\n"
        md += "  calls_these_skills: []\n"
        md += "  called_by_these_skills: []\n"
        md += "```\n\n"
        
        md += "## Error Handling\n\n"
        md += "```yaml\n"
        md += "error_scenarios:\n"
        md += "  data_unavailable:\n"
        md += "    condition: \"Required data not accessible\"\n"
        md += "    resolution: \"Use cached data with caveat\"\n"
        md += "    escalate_if: \"Cache also unavailable\"\n"
        md += "```\n\n"
        
        md += "## Examples\n\n"
        md += "See examples.yaml for detailed workflows\n\n"

        # Generate script filename from intention
        script_name = self._to_snake_case(intention) + ".py"

        md += "## Script Reference\n\n"
        md += f"This skill uses the following Python script for execution:\n\n"
        md += f"```yaml\n"
        md += f"script:\n"
        md += f"  path: scripts/{script_name}\n"
        md += f"  description: Main execution script for {intention.replace('-', ' ')}\n"
        md += f"  usage: Import and call the main function to execute this skill\n"
        md += f"```\n\n"
        md += f"**To execute this skill programmatically:**\n"
        md += f"```python\n"
        md += f"from scripts.{self._to_snake_case(intention)} import *\n"
        md += f"```\n\n"

        md += "## Configuration\n\n"
        md += "This skill can be customized via `config.yaml`. Key settings include:\n\n"
        md += "```yaml\n"
        md += "# Business Rules\n"
        md += "business_rules:\n"
        md += "  approval_required: false\n"
        md += "  approval_threshold: 1000\n"
        md += "  enable_notifications: true\n"
        md += "\n"
        md += "# Feature Flags\n"
        md += "feature_flags:\n"
        md += "  enable_audit_logging: true\n"
        md += "  enable_dry_run_mode: false\n"
        md += "```\n\n"
        md += "Edit `config.yaml` to customize thresholds, integrations, and notifications for your business.\n\n"

        md += "## Additional Resources\n\n"
        md += f"- scripts/{script_name} - Main execution script\n"
        md += "- config.yaml - **Customizable settings for your business**\n"
        md += "- metadata.json - Skill metadata and configuration\n"
        md += "- rules.yaml - Business rules and constraints\n"
        md += "- definitions.yaml - Term definitions and glossary\n"
        md += "- examples.yaml - Detailed workflow examples\n"

        return md
    
    def _generate_natural_triggers(
        self,
        intention: str,
        knowledge: List[ExtractedKnowledge]
    ) -> List[str]:
        """Generate NATURAL LANGUAGE triggers matching customer speech"""
        
        prompt = f"""Generate 5-7 NATURAL LANGUAGE triggers.

INTENTION: {intention}

CRITICAL REQUIREMENTS:
- Match how REAL customers speak
- NO robotic phrases
- NO keywords or technical terms
- Use pattern matching with variables: {{product}}, {{category}}, {{quantity}}
- Must be full natural thoughts

CUSTOMER LANGUAGE TEMPLATES:
- Questions: "Where is/are...", "Do you have...", "Can I get..."
- Requests: "I need...", "I'm looking for...", "I want to..."
- Problems: "This isn't working...", "I'd like to return...", "There's an issue with..."

WRONG (robotic):
- "assist product location"
- "help with assist product location"

RIGHT (natural):
- "where (can I|do I) find {{product}}"
- "do you (have|carry|sell) {{product}}"
- "(I'm|I am) looking for {{product}}"

Return 5-7 triggers, one per line, starting with "TRIGGER:"
"""
        
        try:
            response = self.client.invoke(prompt, max_tokens=800)
            
            triggers = []
            for line in response.split('\n'):
                if line.strip().startswith('TRIGGER:'):
                    trigger = line.replace('TRIGGER:', '').strip()
                    if len(trigger) > 10:
                        triggers.append(trigger)
            
            if not triggers:
                triggers = [
                    f"(I need|I want) to {intention.replace('-', ' ')}",
                    f"(Can you|Could you) help me {intention.replace('-', ' ')}",
                    f"(I'm|I am) trying to {intention.replace('-', ' ')}"
                ]
            
            return triggers[:7]
            
        except:
            return [
                f"(I need|I want) to {intention.replace('-', ' ')}",
                f"(Can you|Could you) help me {intention.replace('-', ' ')}",
                f"(I'm|I am) trying to {intention.replace('-', ' ')}"
            ][:7]
    
    def _extract_steps(self, content: str) -> str:
        """Extract steps"""
        numbered = re.findall(r'(?:^|\n)\s*\d+[\.)]\s*(.+?)(?=\n\s*\d+[\.)]|\n\n|$)', content, re.DOTALL)
        if numbered and len(numbered) >= 2:
            result = ""
            for i, step in enumerate(numbered, 1):
                result += f"{i}. {step.strip()}\n"
            return result
        
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10][:5]
        if sentences:
            result = ""
            for i, sentence in enumerate(sentences, 1):
                result += f"{i}. {sentence}\n"
            return result
        
        return "1. Execute workflow\n"
    
    def _generate_production_rules_yaml(
        self,
        decision_rules: List[ExtractedKnowledge],
        all_knowledge: List[ExtractedKnowledge]
    ) -> str:
        """Generate production-quality rules.yaml"""
        
        rules_dict = {"rules": []}
        
        for i, rule in enumerate(decision_rules, 1):
            content = rule.content.strip()
            
            # Extract signal word
            signal_word = None
            if "must" in content.lower(): signal_word = "MUST"
            elif "never" in content.lower(): signal_word = "NEVER"
            elif "always" in content.lower(): signal_word = "ALWAYS"
            elif "required" in content.lower(): signal_word = "REQUIRED"
            elif "should" in content.lower(): signal_word = "SHOULD"
            
            rule_entry = {
                "rule_name": f"rule_{i}",
                "rule": content,
                "signal_words": signal_word or "POLICY",
                "authority": "Manager override only",
                "sop_reference": f"{rule.source_section}: {rule.metadata.get('sop_quote', 'See source document')}" if rule.metadata else rule.source_section,
                "violation_severity": "high" if signal_word in ["MUST", "NEVER"] else "medium"
            }
            
            rules_dict["rules"].append(rule_entry)
        
        if not rules_dict["rules"]:
            rules_dict["rules"].append({
                "rule_name": "no_rules_extracted",
                "rule": "No explicit constraints documented in source",
                "signal_words": "INFO",
                "authority": "N/A",
                "sop_reference": "No SOP section found",
                "violation_severity": "info"
            })
        
        return yaml.dump(rules_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def _generate_definitions_yaml(self, definitions: List[ExtractedKnowledge]) -> str:
        """Generate definitions.yaml"""
        
        defs_dict = {"definitions": []}
        
        for defn in definitions:
            content = defn.content.strip()
            
            match = re.match(r'^([^:\n]+?):\s*(.+)', content, re.DOTALL)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
            else:
                lines = content.split('\n')
                term = lines[0].strip()
                definition = '\n'.join(lines[1:]).strip() if len(lines) > 1 else content
            
            defs_dict["definitions"].append({
                "term": term,
                "definition": definition,
                "usage_context": "When applicable"
            })
        
        return yaml.dump(defs_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def _generate_production_examples_yaml(
        self,
        all_knowledge: List[ExtractedKnowledge],
        intention: str,
        procedures: List[ExtractedKnowledge]
    ) -> str:
        """Generate REALISTIC examples with full workflows"""
        
        examples_dict = {"examples": []}
        
        # Example 1: Normal flow
        examples_dict["examples"].append({
            "scenario": "Normal successful execution",
            "customer_input": f"I need to {intention.replace('-', ' ')}",
            "agent_workflow": [
                {
                    "step": "Parse request",
                    "result": {"intent": intention, "confidence": 0.95}
                },
                {
                    "step": "Verify preconditions",
                    "action": "Check user authorization",
                    "result": "Authorized"
                },
                {
                    "step": "Execute workflow",
                    "action": "Follow documented procedure",
                    "result": "Completed"
                },
                {
                    "step": "Validate outcome",
                    "result": "Success"
                }
            ],
            "skills_chained": [intention],
            "expected_outcome": "Task completed successfully",
            "test_data_used": ["USER-001", "SESSION-123"]
        })
        
        # Example 2: Error scenario
        examples_dict["examples"].append({
            "scenario": "Data unavailable error",
            "customer_input": f"Can you help me {intention.replace('-', ' ')}",
            "agent_workflow": [
                {
                    "step": "Parse request",
                    "result": {"intent": intention, "confidence": 0.92}
                },
                {
                    "step": "Attempt data retrieval",
                    "action": "Query primary data source",
                    "result": "ERROR: Database unavailable"
                },
                {
                    "step": "Use fallback",
                    "action": "Use cached data with caveat",
                    "result": "Partial data available"
                },
                {
                    "step": "Communicate limitation",
                    "script": "I can help with that, but our system is currently updating. Let me use our backup data.",
                    "result": "Customer informed"
                }
            ],
            "skills_chained": [intention],
            "expected_outcome": "Completed with caveat",
            "test_data_used": ["USER-002", "CACHE-789"]
        })
        
        return yaml.dump(examples_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _generate_config_yaml(
        self,
        intention: str,
        decision_rules: List[ExtractedKnowledge],
        domain: str,
        enterprise_name: Optional[str]
    ) -> str:
        """Generate config.yaml for skill customization.

        This configuration file allows users to customize skill behavior
        for their specific business requirements.
        """

        # Extract configurable parameters from rules
        thresholds = []
        limits = []

        for rule in decision_rules:
            content = rule.content.lower()
            # Look for numeric values that could be configurable
            import re
            numbers = re.findall(r'\b(\d+)\s*(days?|hours?|minutes?|items?|percent|%|dollars?|\$|units?)\b', content, re.IGNORECASE)
            for num, unit in numbers:
                param_name = f"{unit.rstrip('s')}_threshold"
                thresholds.append({
                    "name": param_name,
                    "value": int(num),
                    "unit": unit,
                    "source_rule": rule.content[:100]
                })

        # Build config structure
        config_dict = {
            "# Configuration file for": intention,
            "# Customize these settings for your business requirements": None,
            "# Generated from": domain,
            "skill_config": {
                "name": intention,
                "version": "1.0.0",
                "enabled": True,
                "enterprise": enterprise_name or domain,
                "domain": domain
            },
            "execution_settings": {
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "retry_delay_seconds": 5,
                "async_execution": False,
                "logging_level": "INFO"
            },
            "business_rules": {
                "# Customize thresholds and limits for your business": None,
                "approval_required": False,
                "approval_threshold": 1000,
                "max_items_per_request": 100,
                "enable_notifications": True,
                "escalation_enabled": True
            },
            "integration_settings": {
                "# Configure external system connections": None,
                "database": {
                    "enabled": True,
                    "connection_pool_size": 5,
                    "query_timeout_seconds": 30
                },
                "api": {
                    "enabled": True,
                    "base_url": "",
                    "api_key_env_var": "API_KEY",
                    "rate_limit_per_minute": 60
                },
                "cache": {
                    "enabled": True,
                    "ttl_seconds": 3600,
                    "max_size_mb": 100
                }
            },
            "notification_settings": {
                "# Configure how notifications are sent": None,
                "email": {
                    "enabled": False,
                    "recipients": [],
                    "on_success": False,
                    "on_failure": True,
                    "on_escalation": True
                },
                "slack": {
                    "enabled": False,
                    "webhook_url_env_var": "SLACK_WEBHOOK_URL",
                    "channel": "",
                    "on_success": False,
                    "on_failure": True
                }
            },
            "custom_parameters": {
                "# Add your custom business parameters here": None,
                "param1": {
                    "value": "",
                    "description": "Custom parameter 1"
                },
                "param2": {
                    "value": "",
                    "description": "Custom parameter 2"
                }
            },
            "feature_flags": {
                "# Enable or disable specific features": None,
                "enable_audit_logging": True,
                "enable_detailed_errors": False,
                "enable_performance_metrics": True,
                "enable_dry_run_mode": False,
                "enable_debug_mode": False
            }
        }

        # Add extracted thresholds if any were found
        if thresholds:
            config_dict["extracted_thresholds"] = {
                "# These values were extracted from your source documents": None,
                "# Review and adjust as needed": None,
                "thresholds": thresholds[:5]  # Limit to 5
            }

        # Custom YAML dump that handles None values as comments
        def custom_dump(data, indent=0):
            lines = []
            prefix = "  " * indent

            for key, value in data.items():
                if key.startswith("#"):
                    lines.append(f"{prefix}{key}")
                elif value is None:
                    continue
                elif isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(custom_dump(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"{prefix}  -")
                            for k, v in item.items():
                                lines.append(f"{prefix}    {k}: {repr(v) if isinstance(v, str) else v}")
                        else:
                            lines.append(f"{prefix}  - {item}")
                elif isinstance(value, bool):
                    lines.append(f"{prefix}{key}: {str(value).lower()}")
                elif isinstance(value, str):
                    if value:
                        lines.append(f"{prefix}{key}: \"{value}\"")
                    else:
                        lines.append(f"{prefix}{key}: \"\"")
                else:
                    lines.append(f"{prefix}{key}: {value}")

            return lines

        yaml_lines = custom_dump(config_dict)
        return "\n".join(yaml_lines)

    def _create_fallback_skill(
        self, 
        enterprise_name: Optional[str], 
        knowledge: List[ExtractedKnowledge] = None
    ) -> List[Dict[str, Any]]:
        """Create fallback skill"""
        
        intention = "execute-workflow"
        
        skill_md = f"""---
name: {intention}
description: Execute documented workflow
version: 1.0.0
triggers:
  - pattern: "(I need|I want) to execute (the|a) workflow"
  - pattern: "(Can you|Could you) help me (with|execute) (the|a) workflow"
---

## Overview
Executes workflows per documentation.

## When to Use
When customer requests workflow execution.

## Instructions
1. Verify intent
2. Execute steps
3. Validate outcome

## Rules
- **MUST**: Follow documented procedures
- **NEVER**: Skip validation steps

## Examples
See examples.yaml

## Additional Resources
- metadata.json
"""
        
        metadata = SkillMetadata(
            name=intention,
            description="Execute workflow",
            version="1.0.0",
            source_documents=["fallback"],
            confidence=0.5,
            created_date=datetime.now().isoformat(),
            enterprise=enterprise_name,
            domain=enterprise_name or "Enterprise",
            knowledge_extraction_stats={}
        )
        
        # Generate default config for fallback
        fallback_config = self._generate_config_yaml(intention, [], enterprise_name or "Enterprise", enterprise_name)

        return [{
            "metadata": asdict(metadata),
            "skill_md": skill_md,
            "rules_yaml": yaml.dump({"rules": []}, default_flow_style=False),
            "definitions_yaml": yaml.dump({"definitions": []}, default_flow_style=False),
            "examples_yaml": yaml.dump({"examples": []}, default_flow_style=False),
            "config_yaml": fallback_config,
            "script_py": None
        }]

    # ==========================================
    # SCRIPT GENERATION METHODS
    # ==========================================

    def _generate_script_for_skill(self, skill_name: str, knowledge: List[ExtractedKnowledge], domain: str) -> Optional[str]:
        """Generate a Python script for a skill with retry logic."""
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    logger.info(f"   🐍 Generating Python script...")
                else:
                    logger.info(f"   🔄 Retry attempt {attempt + 1}/{max_retries}...")

                # Generate script components using LLM
                if attempt == 0:
                    components = self._generate_script_components(skill_name, knowledge)
                else:
                    components = self._generate_script_components_with_retry(skill_name, knowledge, last_error, attempt)

                # Validate components
                validated = self._validate_script_components(components)

                # Sanitize description for module docstring (no triple quotes or newlines)
                safe_description = validated['description'][:100].replace('"""', "'").replace("'''", "'").replace('\n', ' ')

                # Fill template
                script = SCRIPT_TEMPLATE.format(
                    skill_name=skill_name,
                    skill_description=safe_description,
                    knowledge_source=domain,
                    version="1.0.0",
                    class_name=self._to_class_name(skill_name),
                    class_docstring=validated['class_docstring'],
                    initialization_code=validated['init_code'],
                    input_description=validated['input_desc'],
                    return_description=validated['return_desc'],
                    main_execution_logic=validated['main_logic'],
                    validation_logic=validated['validation'],
                    helper_methods=validated['helpers'],
                    required_inputs=json.dumps(validated['required_inputs']),
                    optional_inputs=json.dumps(validated['optional_inputs']),
                    outputs=json.dumps(validated['outputs'])
                )

                # Debug: Show first 20 lines if there's an issue
                logger.debug(f"   Generated script preview:\n" + '\n'.join(f"{i+1}: {line}" for i, line in enumerate(script.split('\n')[:20])))

                # Validate syntax
                self._validate_python_syntax(script, skill_name)

                if attempt > 0:
                    logger.info(f"   ✅ Script succeeded on retry #{attempt}")
                else:
                    logger.info(f"   ✅ Script generated and validated")
                return script

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning(f"   ⚠️ Attempt {attempt + 1} failed: {last_error}")
                else:
                    logger.error(f"   ❌ Script generation failed after {max_retries} attempts: {last_error}")

        return None

    def _generate_script_components(self, skill_name: str, knowledge: List[ExtractedKnowledge]) -> Dict:
        """Use LLM to generate script components."""

        # Build knowledge summary
        knowledge_summary = []
        for k in knowledge[:5]:  # Limit to avoid token overflow
            knowledge_summary.append({
                "type": k.type,
                "content": k.content[:500] if len(k.content) > 500 else k.content
            })

        prompt = f"""Generate ONLY the following components for a skill script based on this knowledge.

Skill Name: {skill_name}

Knowledge:
{json.dumps(knowledge_summary, indent=2)}

Generate as JSON with these exact keys:
{{
  "description": "Brief description of the skill",
  "class_docstring": "Clear 2-3 sentence description of what this skill does",
  "init_code": "Python code for initialization (no imports, just logic like self.threshold = 100)",
  "input_desc": "Description of expected input_data structure (e.g., 'Dict with keys: request_type, amount')",
  "return_desc": "Description of return value structure (e.g., 'Dict with keys: approved, reason')",
  "main_logic": "Python code for main execution logic (4-12 lines, MUST set 'result' variable)",
  "validation": "Python code to validate input_data (2-6 lines, MUST return True/False)",
  "helpers": "Optional helper method definitions (complete method defs with 'def method_name(self, ...):')",
  "required_inputs": ["list", "of", "required", "keys"],
  "optional_inputs": ["list", "of", "optional", "keys"],
  "outputs": ["list", "of", "output", "keys"]
}}

CRITICAL SYNTAX RULES (AVOID COMMON ERRORS):
1. Use 'if' NOT 'iif' or 'iff' (typo - spell it correctly!)
2. Every if/else/for/while/try/except block MUST have body
   - Empty blocks? Add 'pass'
   - Example: if x > 10:\n    pass
3. main_logic MUST set 'result' variable (a dict)
   - Example: result = {{"success": True, "data": processed}}
4. validation MUST have 'return' statement
   - Example: return True
5. No import statements in any code blocks
6. No eval/exec/os.system/__import__ or dangerous functions
7. Keep logic concise and focused on the skill's purpose
8. Use only standard Python operations

Return ONLY valid JSON, no markdown or explanation."""

        response = self.client.invoke(prompt, max_tokens=2000)
        response_clean = self._clean_json_response(response)
        return json.loads(response_clean)

    def _generate_script_components_with_retry(self, skill_name: str, knowledge: List[ExtractedKnowledge], previous_error: str, attempt: int) -> Dict:
        """Generate script components with error feedback from previous attempt."""

        # Build knowledge summary
        knowledge_summary = []
        for k in knowledge[:5]:
            knowledge_summary.append({
                "type": k.type,
                "content": k.content[:500] if len(k.content) > 500 else k.content
            })

        prompt = f"""Generate ONLY the following components for a skill script based on this knowledge.

Skill Name: {skill_name}

Knowledge:
{json.dumps(knowledge_summary, indent=2)}

IMPORTANT: Previous attempt failed with error: {previous_error}

Common mistakes to avoid:
1. Typo 'iif' instead of 'if'
2. Typo 'iff' instead of 'if'
3. Empty blocks after if/else/for/while (add 'pass' if empty)
4. Using dangerous functions: eval, exec, __import__, os.system
5. Not setting 'result' variable in main_logic
6. Not having 'return' statement in validation

Generate as JSON with these exact keys:
{{
  "description": "Brief description of the skill",
  "class_docstring": "Clear 2-3 sentence description of what this skill does",
  "init_code": "Python code for initialization (no imports, just logic like self.threshold = 100)",
  "input_desc": "Description of expected input_data structure",
  "return_desc": "Description of return value structure",
  "main_logic": "Python code for main execution logic (4-12 lines, MUST set 'result' variable)",
  "validation": "Python code to validate input_data (2-6 lines, MUST return True/False)",
  "helpers": "Optional helper method definitions (complete method defs)",
  "required_inputs": ["list", "of", "required", "keys"],
  "optional_inputs": ["list", "of", "optional", "keys"],
  "outputs": ["list", "of", "output", "keys"]
}}

CRITICAL RULES:
- Use 'if' not 'iif' or 'iff'
- Every if/else/for/while block must have at least one statement (use 'pass' if needed)
- main_logic MUST set 'result' variable (a dict)
- validation MUST return True or False
- No eval/exec/os.system/__import__
- No import statements
- Keep logic simple and focused

Return ONLY valid JSON, no markdown or explanation."""

        response = self.client.invoke(prompt, max_tokens=2000)
        response_clean = self._clean_json_response(response)
        return json.loads(response_clean)

    def _validate_script_components(self, components: Dict) -> Dict:
        """Validate and format script components."""
        validated = {}

        # Check dangerous patterns
        dangerous = ['eval', 'exec', '__import__', 'os.system', 'subprocess']
        for key, value in components.items():
            if any(p in str(value) for p in dangerous):
                raise ValueError(f"Dangerous pattern in {key}")

        # Required fields
        required = ['main_logic', 'validation', 'class_docstring']
        for req in required:
            if req not in components:
                raise ValueError(f"Missing: {req}")

        # Sanitize string components (remove triple quotes, limit to single line for docstrings)
        def sanitize_docstring(s):
            if not s:
                return "No description provided"
            # Remove triple quotes that would break the template
            s = s.replace('"""', "'")
            s = s.replace("'''", "'")
            # Replace newlines with spaces for docstrings (keep it on one logical line)
            s = ' '.join(s.split())
            return s

        # Pre-validate code snippets before indentation
        def validate_code_snippet(code, name):
            """Check if code snippet has valid Python syntax."""
            if not code or not code.strip():
                return 'pass'
            # Strip markdown code blocks if present
            code = code.strip()
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            code = code.strip()
            # Wrap in a function to make it valid Python
            test_code = f"def _test():\n" + '\n'.join('    ' + line for line in code.split('\n'))
            try:
                compile(test_code, f'<{name}>', 'exec')
            except SyntaxError as e:
                logger.warning(f"   Code snippet '{name}' has syntax error: {e.msg}")
                logger.warning(f"   Problematic code: {code[:100]}...")
                # Return a safe default
                if name == 'main_logic':
                    return 'result = {"success": True, "message": "Default response"}'
                elif name == 'validation':
                    return 'return True'
                else:
                    return 'pass'
            return code

        init_code = validate_code_snippet(components.get('init_code', 'pass'), 'init_code')
        main_logic = validate_code_snippet(components.get('main_logic', 'result = {"success": True}'), 'main_logic')
        validation_code = validate_code_snippet(components.get('validation', 'return True'), 'validation')

        # Process with indentation
        validated['init_code'] = self._indent_code(init_code, 8)
        validated['main_logic'] = self._indent_code(main_logic, 12)
        validated['validation'] = self._indent_code(validation_code, 8)

        # Validate and process helpers (most likely source of syntax errors)
        helpers = components.get('helpers', '')
        if helpers and helpers.strip():
            # Strip markdown code blocks
            helpers = helpers.strip()
            if helpers.startswith('```python'):
                helpers = helpers[9:]
            elif helpers.startswith('```'):
                helpers = helpers[3:]
            if helpers.endswith('```'):
                helpers = helpers[:-3]
            helpers = helpers.strip()

            # Try to validate the helper methods
            try:
                # Wrap in a class to validate
                test_code = f"class _Test:\n" + '\n'.join('    ' + line for line in helpers.split('\n'))
                compile(test_code, '<helpers>', 'exec')
                validated['helpers'] = '\n' + self._indent_code(helpers, 4)
            except SyntaxError as e:
                logger.warning(f"   Helper methods have syntax error: {e.msg}")
                logger.warning(f"   Helpers code: {helpers[:100]}...")
                validated['helpers'] = ''  # Skip bad helpers
        else:
            validated['helpers'] = ''

        validated['description'] = sanitize_docstring(components.get('description', 'No description'))
        validated['class_docstring'] = sanitize_docstring(components['class_docstring'])
        validated['input_desc'] = sanitize_docstring(components.get('input_desc', 'Dict with input data'))
        validated['return_desc'] = sanitize_docstring(components.get('return_desc', 'Dict with results'))
        validated['required_inputs'] = components.get('required_inputs', [])
        validated['optional_inputs'] = components.get('optional_inputs', [])
        validated['outputs'] = components.get('outputs', ['result'])

        return validated

    @staticmethod
    def _indent_code(code: str, spaces: int) -> str:
        """Add proper indentation preserving relative indent."""
        if not code or not code.strip():
            return ' ' * spaces + 'pass'

        lines = code.split('\n')

        # Find minimum indentation
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                leading = len(line) - len(line.lstrip())
                min_indent = min(min_indent, leading)

        if min_indent == float('inf'):
            min_indent = 0

        indented = []
        for line in lines:
            if line.strip():
                leading = len(line) - len(line.lstrip())
                relative = leading - min_indent
                indented.append(' ' * spaces + ' ' * relative + line.lstrip())
            else:
                indented.append('')

        return '\n'.join(indented)

    @staticmethod
    def _validate_python_syntax(code: str, name: str) -> None:
        """Validate Python syntax."""
        try:
            compile(code, f"<{name}>", 'exec')
        except SyntaxError as e:
            # Show the problematic line for debugging
            lines = code.split('\n')
            error_line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else "N/A"
            logger.debug(f"   Syntax error context - Line {e.lineno}: {error_line[:100]}")
            raise SyntaxError(f"Syntax error at line {e.lineno}: {e.msg} - '{error_line[:50]}...'")

    @staticmethod
    def _to_class_name(text: str) -> str:
        """Convert to PascalCase class name (no hyphens allowed in Python class names)."""
        # Replace hyphens and underscores with spaces, then split
        text = text.replace('-', ' ').replace('_', ' ')
        # Remove non-alphanumeric except spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Split on whitespace and capitalize each word
        words = text.split()
        return ''.join(word.capitalize() for word in words) + 'Skill'

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """Convert to snake_case."""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[\s-]+', '_', text)
        return text.strip('_')

    def _complete_outputs(self, skills: List[Dict[str, Any]], output_dir: str):
        """PHASE 5: Save skills and scripts

        Output structure (matches .claude/skills/ format):
        output_dir/skills/
        ├── skill_name/
        │   ├── scripts/
        │   │   └── skill_name.py
        │   ├── SKILL.md
        │   ├── config.yaml
        │   ├── metadata.json
        │   ├── rules.yaml
        │   ├── definitions.yaml
        │   └── examples.yaml
        """

        # Create skills directory inside output_dir
        skills_dir = os.path.join(output_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)

        scripts_count = 0

        for i, skill in enumerate(skills, 1):
            skill_name = skill["metadata"]["name"]
            skill_dir = os.path.join(skills_dir, skill_name)
            os.makedirs(skill_dir, exist_ok=True)

            logger.info(f"💾 [{i}/{len(skills)}] Saving: {skill_name}")

            # Save all skill files
            files_to_save = {
                "SKILL.md": skill["skill_md"],
                "config.yaml": skill.get("config_yaml", ""),
                "metadata.json": json.dumps(skill["metadata"], indent=2),
                "rules.yaml": skill["rules_yaml"],
                "definitions.yaml": skill["definitions_yaml"],
                "examples.yaml": skill["examples_yaml"]
            }

            for filename, content in files_to_save.items():
                try:
                    filepath = os.path.join(skill_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"   ✅ {filename}")
                except Exception as e:
                    logger.error(f"   ❌ {filename}: {e}")

            # Save Python script INSIDE skill folder under scripts/ subdirectory
            script_py = skill.get("script_py")
            if script_py:
                try:
                    # Create scripts/ folder inside skill folder
                    script_dir = os.path.join(skill_dir, "scripts")
                    os.makedirs(script_dir, exist_ok=True)

                    script_filename = self._to_snake_case(skill_name) + ".py"
                    script_path = os.path.join(script_dir, script_filename)

                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(script_py)

                    logger.info(f"   🐍 Script: scripts/{script_filename}")
                    scripts_count += 1
                except Exception as e:
                    logger.error(f"   ❌ Script failed: {e}")

            logger.info(f"   ✅ Complete")

        logger.info(f"📊 Total: {len(skills)} skills, {scripts_count} scripts")
        logger.info(f"📁 Output: {skills_dir}")
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response.

        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra text before/after the JSON object
        - Nested braces in the JSON
        """
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Find the first { which starts the JSON object
        start_idx = response.find('{')
        if start_idx == -1:
            logger.warning("No JSON object found in response")
            return response

        # Find the matching closing brace by tracking depth
        depth = 0
        in_string = False
        escape_next = False
        end_idx = start_idx

        for i, char in enumerate(response[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if depth != 0:
            logger.warning(f"Unbalanced braces in JSON (depth={depth}), returning original")
            return response

        # Extract just the JSON object
        json_str = response[start_idx:end_idx + 1]

        if end_idx + 1 < len(response):
            extra = response[end_idx + 1:].strip()
            if extra:
                logger.info(f"   Trimmed {len(extra)} chars of extra text after JSON")

        return json_str


# ============================================================================
# LAMBDA HANDLER - Async processing with CORS support
# ============================================================================

def process_extraction(file_content: bytes, filename: str, enterprise_name: Optional[str], job_id: str):
    """Background processing function"""
    try:
        logger.info(f"[{job_id}] Processing file: {filename}")
        logger.info(f"[{job_id}] File size: {len(file_content)} bytes")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # Parse document to text
        text_content = parse_document(tmp_path, filename)
        logger.info(f"[{job_id}] Extracted text: {len(text_content)} characters")
        
        # Save as temporary text file for orchestrator
        text_file_path = tmp_path + '.txt'
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Run knowledge extraction
        output_dir = tempfile.mkdtemp()
        
        orchestrator = KnowledgeCaptureOrchestrator(region=os.environ.get('AWS_REGION', 'us-west-2'))
        result = orchestrator.process_documents(
            document_paths=[text_file_path],
            enterprise_name=enterprise_name,
            output_dir=output_dir
        )
        
        # Upload skills to S3
        s3_paths = []
        logger.info(f"[{job_id}] Uploading files from: {output_dir}")
        logger.info(f"[{job_id}] Target bucket: {SKILLS_BUCKET}")

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, output_dir)
                # Ensure forward slashes for S3 keys (important for cross-platform compatibility)
                relative_path = relative_path.replace('\\', '/')
                s3_key = f"Knowledge_Extraction_To_Skills/{job_id}/{relative_path}"

                logger.info(f"[{job_id}] Uploading: {local_path} -> s3://{SKILLS_BUCKET}/{s3_key}")
                get_s3_client().upload_file(local_path, SKILLS_BUCKET, s3_key)
                s3_paths.append(s3_key)
                logger.info(f"[{job_id}] ✅ Uploaded: {s3_key}")

        logger.info(f"[{job_id}] Total files uploaded: {len(s3_paths)}")
        
        # Save completion status
        completion_result = {
            'status': 'completed',
            'message': 'Skills extracted successfully',
            'result': result,
            's3_bucket': SKILLS_BUCKET,
            's3_paths': s3_paths,
            'job_id': job_id,
            'completed_at': datetime.now().isoformat()
        }
        
        get_s3_client().put_object(
            Bucket=SKILLS_BUCKET,
            Key=f"Knowledge_Extraction_To_Skills/{job_id}/status.json",
            Body=json.dumps(completion_result, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"[{job_id}] ✅ Processing complete")
        
        # Cleanup
        os.unlink(tmp_path)
        os.unlink(text_file_path)
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Processing failed: {str(e)}", exc_info=True)
        
        # Save error status
        error_result = {
            'status': 'failed',
            'error': str(e),
            'type': type(e).__name__,
            'job_id': job_id,
            'failed_at': datetime.now().isoformat()
        }
        
        get_s3_client().put_object(
            Bucket=SKILLS_BUCKET,
            Key=f"Knowledge_Extraction_To_Skills/{job_id}/status.json",
            Body=json.dumps(error_result, indent=2),
            ContentType='application/json'
        )


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    UPDATED LAMBDA HANDLER - Supports BOTH upload methods:
    1. Multipart file upload (POST with multipart/form-data)
    2. Base64 JSON upload (POST with application/json)
    3. Status check (GET /status/{job_id})
    
    Expected request body for base64 JSON:
    {
        "file": "base64_encoded_file_content",
        "filename": "document.pdf",
        "enterprise_name": "RetailCo" (optional)
    }
    
    Expected multipart form-data:
    - file: [binary file]
    - enterprise_name: "RetailCo" (optional)
    """
    
    try:
        # Handle OPTIONS request (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': CORS_HEADERS,
                'body': json.dumps({'message': 'CORS preflight'})
            }
        
        # Handle GET - Status check
        if event.get('httpMethod') == 'GET':
            if event.get('pathParameters') and 'job_id' in event['pathParameters']:
                job_id = event['pathParameters']['job_id']
                
                try:
                    response = get_s3_client().get_object(
                        Bucket=SKILLS_BUCKET,
                        Key=f"Knowledge_Extraction_To_Skills/{job_id}/status.json"
                    )
                    status_data = json.loads(response['Body'].read().decode('utf-8'))
                    
                    return {
                        'statusCode': 200,
                        'headers': CORS_HEADERS,
                        'body': json.dumps(status_data)
                    }
                except get_s3_client().exceptions.NoSuchKey:
                    return {
                        'statusCode': 404,
                        'headers': CORS_HEADERS,
                        'body': json.dumps({
                            'error': 'Job not found',
                            'job_id': job_id
                        })
                    }
        
        # Check if this is a background invocation
        if event.get('is_background_job'):
            # Background processing (no CORS needed for internal calls)
            job_data = event['job_data']
            file_content = base64.b64decode(job_data['file'])
            process_extraction(
                file_content=file_content,
                filename=job_data['filename'],
                enterprise_name=job_data.get('enterprise_name'),
                job_id=job_data['job_id']
            )
            return {'statusCode': 200, 'body': 'Background processing complete'}
        
        # Handle POST - File upload
        if event.get('httpMethod') == 'POST':
            content_type = event.get('headers', {}).get('Content-Type') or \
                          event.get('headers', {}).get('content-type', '')
            
            # Method 1: Multipart file upload
            if 'multipart/form-data' in content_type:
                logger.info("Processing multipart/form-data upload")
                
                body = event.get('body', '')
                is_base64 = event.get('isBase64Encoded', False)
                
                if not is_base64:
                    # API Gateway should base64 encode binary data
                    # If not, we need to handle it
                    body = base64.b64encode(body.encode()).decode()
                
                parsed_data = parse_multipart_form_data(body, content_type)
                
                file_content = parsed_data.get('file')
                filename = parsed_data.get('filename')
                enterprise_name = parsed_data.get('enterprise_name')
                
                if not file_content or not filename:
                    return {
                        'statusCode': 400,
                        'headers': CORS_HEADERS,
                        'body': json.dumps({
                            'error': 'Missing file or filename in multipart upload'
                        })
                    }
                
                # Convert to base64 for internal processing
                file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            # Method 2: Base64 JSON upload (your current method)
            elif 'application/json' in content_type:
                logger.info("Processing JSON with base64 file")
                
                body = json.loads(event.get('body', '{}'))
                
                if 'file' not in body or 'filename' not in body:
                    return {
                        'statusCode': 400,
                        'headers': CORS_HEADERS,
                        'body': json.dumps({
                            'error': 'Missing required fields: file, filename',
                            'hint': 'Send either multipart/form-data or JSON with base64-encoded file'
                        })
                    }
                
                file_base64 = body['file']
                filename = body['filename']
                enterprise_name = body.get('enterprise_name')
            
            else:
                return {
                    'statusCode': 400,
                    'headers': CORS_HEADERS,
                    'body': json.dumps({
                        'error': 'Invalid Content-Type',
                        'expected': 'multipart/form-data or application/json',
                        'received': content_type
                    })
                }
            
            # Generate unique job ID
            job_id = f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"
            
            # Save initial status
            initial_status = {
                'status': 'processing',
                'message': 'Extraction job started',
                'job_id': job_id,
                'filename': filename,
                'enterprise_name': enterprise_name,
                'started_at': datetime.now().isoformat(),
                'estimated_completion': '5-10 minutes'
            }
            
            get_s3_client().put_object(
                Bucket=SKILLS_BUCKET,
                Key=f"Knowledge_Extraction_To_Skills/{job_id}/status.json",
                Body=json.dumps(initial_status, indent=2),
                ContentType='application/json'
            )
            
            # Invoke Lambda asynchronously
            lambda_client = boto3.client('lambda')
            lambda_client.invoke(
                FunctionName=context.function_name,
                InvocationType='Event',  # Async invocation
                Payload=json.dumps({
                    'is_background_job': True,
                    'job_data': {
                        'file': file_base64,
                        'filename': filename,
                        'enterprise_name': enterprise_name,
                        'job_id': job_id
                    }
                })
            )
            
            logger.info(f"[{job_id}] Async job started")
            
            # Return immediately with CORS headers
            return {
                'statusCode': 202,  # Accepted
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'status': 'accepted',
                    'message': 'Extraction job started',
                    'job_id': job_id,
                    'filename': filename,
                    'check_status_url': f"GET /status/{job_id}",
                    'estimated_completion': '5-10 minutes'
                })
            }
        
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'error': 'Invalid request',
                'supported_methods': ['POST /extract', 'GET /status/{job_id}']
            })
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

        return {
            'statusCode': 500,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }


# ============================================================================
# LOCAL EXECUTION MODE - For testing without AWS Lambda
# ============================================================================

class LocalAnthropicClient:
    """Local client using Anthropic API directly (not Bedrock)"""

    def __init__(self, api_key: str = None):
        try:
            import anthropic
            self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-sonnet-4-20250514"
            logger.info("✅ Anthropic client initialized (local mode)")
        except ImportError:
            raise ImportError("Local mode requires 'anthropic' package. Install with: pip install anthropic")

    def invoke(self, prompt: str, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        """Invoke Claude API directly"""
        if len(prompt) > MAX_INPUT_CHARS:
            logger.warning(f"⚠️  Truncating prompt from {len(prompt)} chars")
            prompt = prompt[:MAX_INPUT_CHARS] + "\n\n[TRUNCATED]"

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text


class LocalKnowledgeOrchestrator(KnowledgeCaptureOrchestrator):
    """Local orchestrator that uses Anthropic API instead of Bedrock"""

    def __init__(self, api_key: str = None):
        logger.info("="*70)
        logger.info("🚀 LOCAL SKILL GENERATOR (Anthropic API)")
        logger.info("="*70)
        self.client = LocalAnthropicClient(api_key=api_key)
        self.model = self.client.model
        logger.info(f"✅ Ready (Model: {self.model})")
        logger.info("")

    def process_with_review(
        self,
        document_paths: List[str],
        enterprise_name: Optional[str] = None,
        output_dir: str = "./extracted_skills",
        auto_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process documents with interactive review step.

        Steps:
        1. Upload file
        2. Extract knowledge
        3. Save extracted_knowledge.json and ask for review
        4. Generate skills and scripts
        5. Validate and save
        """
        logger.info("="*70)
        logger.info("📚 PROCESSING WITH REVIEW")
        logger.info("="*70)
        logger.info(f"Documents: {len(document_paths)}")
        logger.info(f"Enterprise: {enterprise_name or 'Will be inferred'}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Auto mode: {auto_mode}")
        logger.info("")

        try:
            # ==========================================
            # STEP 1: UPLOAD FILE
            # ==========================================
            print("\n" + "="*60)
            print("STEP 1: UPLOAD FILE")
            print("="*60)

            documents = self._upload_documents(document_paths, enterprise_name)
            logger.info(f"✅ Uploaded {len(documents)} document(s)")

            if not documents:
                raise ValueError("No documents were successfully uploaded")

            # ==========================================
            # STEP 2: EXTRACT KNOWLEDGE
            # ==========================================
            print("\n" + "="*60)
            print("STEP 2: EXTRACT KNOWLEDGE")
            print("="*60)

            extracted_knowledge = self._extract_knowledge(documents)
            logger.info(f"✅ Extracted {len(extracted_knowledge)} knowledge items")

            # ==========================================
            # STEP 3: REVIEW EXTRACTED KNOWLEDGE
            # ==========================================
            print("\n" + "="*60)
            print("STEP 3: REVIEW EXTRACTED KNOWLEDGE")
            print("="*60)

            # Save extracted knowledge for review
            os.makedirs(output_dir, exist_ok=True)
            knowledge_file = os.path.join(output_dir, "extracted_knowledge.json")

            # Convert to serializable format
            knowledge_data = {
                "document_metadata": {
                    "files_processed": [d.filename for d in documents],
                    "enterprise": enterprise_name or documents[0].enterprise if documents else "Unknown",
                    "extraction_timestamp": datetime.now().isoformat()
                },
                "extracted_knowledge": [
                    {
                        "type": k.type,
                        "content": k.content,
                        "source_document": k.source_document,
                        "source_section": k.source_section,
                        "confidence": k.confidence,
                        "user_intention": k.user_intention,
                        "metadata": k.metadata
                    }
                    for k in extracted_knowledge
                ],
                "statistics": {
                    "total_items": len(extracted_knowledge),
                    "by_type": {}
                }
            }

            # Count by type
            for k in extracted_knowledge:
                k_type = k.type or "unknown"
                knowledge_data["statistics"]["by_type"][k_type] = \
                    knowledge_data["statistics"]["by_type"].get(k_type, 0) + 1

            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2)

            print(f"\n✅ Extracted knowledge saved to: {knowledge_file}")
            print(f"\n📊 Extraction Statistics:")
            print(f"   - Total items: {len(extracted_knowledge)}")
            for k_type, count in knowledge_data["statistics"]["by_type"].items():
                print(f"   - {k_type}: {count}")

            if not auto_mode:
                print("\n" + "-"*60)
                print("📋 Please review 'extracted_knowledge.json'")
                print("   You can edit the file to modify/remove items before generation.")
                print("-"*60)
                input("\nPress Enter when ready to generate skills and scripts...")

                # Reload in case user edited
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)

                # Reconstruct knowledge objects
                extracted_knowledge = []
                for item in knowledge_data.get("extracted_knowledge", []):
                    k = ExtractedKnowledge(
                        type=item.get("type"),
                        content=item.get("content"),
                        source_document=item.get("source_document", ""),
                        source_section=item.get("source_section"),
                        confidence=item.get("confidence", 0.8),
                        metadata=item.get("metadata", {}),
                        user_intention=item.get("user_intention")
                    )
                    extracted_knowledge.append(k)

                print(f"\n✅ Reloaded {len(extracted_knowledge)} knowledge items after review")

            # Deduplicate
            consolidated = self._review_knowledge(extracted_knowledge)

            # ==========================================
            # STEP 4: GENERATE SKILLS & SCRIPTS
            # ==========================================
            print("\n" + "="*60)
            print("STEP 4: GENERATE SKILLS & SCRIPTS")
            print("="*60)

            skills = self._generate_skills(consolidated, enterprise_name)
            logger.info(f"✅ Generated {len(skills)} skill(s)")

            # ==========================================
            # STEP 5: SAVE & VALIDATE
            # ==========================================
            print("\n" + "="*60)
            print("STEP 5: SAVE & VALIDATE")
            print("="*60)

            self._complete_outputs(skills, output_dir)
            logger.info(f"✅ Saved to {output_dir}")

            # Count scripts
            scripts_count = sum(1 for s in skills if s.get("script_py"))

            result = {
                "status": "success",
                "documents_processed": len(documents),
                "knowledge_items_extracted": len(consolidated),
                "skills_generated": len(skills),
                "scripts_generated": scripts_count,
                "output_directory": output_dir,
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"❌ Failed: {e}", exc_info=True)
            raise


def main():
    """Local execution entry point with interactive review"""
    import sys

    print("="*70)
    print("🏠 LOCAL KNOWLEDGE EXTRACTION PIPELINE")
    print("="*70)
    print("")
    print("Pipeline Steps:")
    print("  1. Upload file (any format: PDF, DOCX, TXT, MD)")
    print("  2. Extract knowledge using Claude API")
    print("  3. Review extracted_knowledge.json (editable)")
    print("  4. Generate SKILL.md files and Python scripts")
    print("  5. Validate scripts and save outputs")
    print("")

    if len(sys.argv) < 2:
        print("Usage: python lambda_handler.py <document_file> [enterprise_name] [--auto]")
        print("")
        print("Arguments:")
        print("  document_file   - Path to SOP/manual file")
        print("  enterprise_name - Optional enterprise name")
        print("  --auto          - Skip review step (auto mode)")
        print("")
        print("Supported formats: .txt, .md, .pdf, .docx")
        print("")
        print("Examples:")
        print("  python lambda_handler.py manual.pdf")
        print("  python lambda_handler.py \"Retail SOP.pdf\" RetailCo")
        print("  python lambda_handler.py document.pdf RetailCo --auto")
        print("")
        print("Environment variables:")
        print("  ANTHROPIC_API_KEY - Required for local execution")
        return 1

    input_file = sys.argv[1]

    # Parse optional arguments
    enterprise_name = None
    auto_mode = "--auto" in sys.argv

    for arg in sys.argv[2:]:
        if arg != "--auto":
            enterprise_name = arg
            break

    # Check file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return 1

    # Check API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("")
        print("Set it with:")
        print("  set ANTHROPIC_API_KEY=your-api-key-here  (Windows)")
        print("  export ANTHROPIC_API_KEY=your-api-key-here  (Linux/Mac)")
        return 1

    try:
        # Create output directory
        output_dir = "./extracted_skills"

        # Initialize local orchestrator
        orchestrator = LocalKnowledgeOrchestrator()

        # Process document with review
        result = orchestrator.process_with_review(
            document_paths=[input_file],
            enterprise_name=enterprise_name,
            output_dir=output_dir,
            auto_mode=auto_mode
        )

        # Summary
        print("")
        print("="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"📁 Generated files:")
        print(f"   - {result.get('skills_generated', 0)} SKILL.md files in '{output_dir}/'")
        print(f"   - {result.get('scripts_generated', 0)} Python scripts in './scripts/'")
        print(f"   - 1 extracted_knowledge.json")
        print("")
        print(f"📂 Output structure:")
        print(f"   {output_dir}/")
        print(f"   ├── extracted_knowledge.json")
        print(f"   ├── skill-name-1/")
        print(f"   │   ├── SKILL.md")
        print(f"   │   ├── metadata.json")
        print(f"   │   ├── rules.yaml")
        print(f"   │   ├── definitions.yaml")
        print(f"   │   └── examples.yaml")
        print(f"   └── ...")
        print(f"   ./scripts/")
        print(f"   ├── skill-name-1/")
        print(f"   │   └── skill_name_1.py")
        print(f"   └── ...")
        print("")
        print("💡 Tips:")
        print("   - To skip review: python lambda_handler.py document.pdf --auto")
        print("   - To test a script: python ./scripts/skill-name/skill_name.py")
        print("")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
