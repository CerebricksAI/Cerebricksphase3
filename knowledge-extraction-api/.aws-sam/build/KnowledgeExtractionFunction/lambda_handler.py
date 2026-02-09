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

# AWS clients
s3 = boto3.client('s3')
SKILLS_BUCKET = os.environ.get('SKILLS_BUCKET')

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

def parse_document(file_path: str, filename: str) -> str:
    """Parse document to plain text based on file extension"""
    ext = Path(filename).suffix.lower()
    
    if ext in ['.txt', '.md']:
        return parse_text(file_path)
    elif ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.docx':
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def parse_text(file_path: str) -> str:
    """Parse plain text files"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_pdf(file_path: str) -> str:
    """Parse PDF files using PyPDF2"""
    import PyPDF2
    text = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    return '\n\n'.join(text)


def parse_docx(file_path: str) -> str:
    """Parse DOCX files using python-docx"""
    from docx import Document
    doc = Document(file_path)
    text = [p.text for p in doc.paragraphs]
    return '\n\n'.join(text)


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
        """PHASE 1: Upload"""
        documents = []
        
        for i, path in enumerate(document_paths, 1):
            try:
                filename = os.path.basename(path)
                logger.info(f"📄 [{i}/{len(document_paths)}] {filename}")
                
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = DocumentContext(
                    filename=filename,
                    file_type=Path(path).suffix.lower(),
                    content=content,
                    enterprise=enterprise_name
                )
                documents.append(doc)
                logger.info(f"   ✅ Uploaded")
                
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
            response = self.client.invoke(prompt, max_tokens=16000)
            response_clean = self._clean_json_response(response)
            
            result = json.loads(response_clean)
            
            inferred_domain = result.get("inferred_domain", "Enterprise Operations")
            logger.info(f"   🏢 Domain: {inferred_domain}")
            doc.enterprise = doc.enterprise or inferred_domain
            
            knowledge_items = []
            for item in result.get("extracted_knowledge", []):
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
            logger.error(f"   ❌ Failed: {e}")
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
            "examples_yaml": examples_yaml
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
        
        md += "## Additional Resources\n\n"
        md += "- metadata.json\n- rules.yaml\n- definitions.yaml\n- examples.yaml\n"
        
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
        
        return [{
            "metadata": asdict(metadata),
            "skill_md": skill_md,
            "rules_yaml": yaml.dump({"rules": []}, default_flow_style=False),
            "definitions_yaml": yaml.dump({"definitions": []}, default_flow_style=False),
            "examples_yaml": yaml.dump({"examples": []}, default_flow_style=False)
        }]
    
    def _complete_outputs(self, skills: List[Dict[str, Any]], output_dir: str):
        """PHASE 5: Save"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, skill in enumerate(skills, 1):
            skill_name = skill["metadata"]["name"]
            skill_dir = os.path.join(output_dir, skill_name)
            os.makedirs(skill_dir, exist_ok=True)
            
            logger.info(f"💾 [{i}/{len(skills)}] Saving: {skill_name}")
            
            # Save all files
            files_to_save = {
                "SKILL.md": skill["skill_md"],
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
            
            logger.info(f"   ✅ Complete")
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON"""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return response.strip()


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
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, output_dir)
                s3_key = f"jobs/{job_id}/{relative_path}"
                
                s3.upload_file(local_path, SKILLS_BUCKET, s3_key)
                s3_paths.append(s3_key)
                logger.info(f"[{job_id}] Uploaded: s3://{SKILLS_BUCKET}/{s3_key}")
        
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
        
        s3.put_object(
            Bucket=SKILLS_BUCKET,
            Key=f"jobs/{job_id}/status.json",
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
        
        s3.put_object(
            Bucket=SKILLS_BUCKET,
            Key=f"jobs/{job_id}/status.json",
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
                    response = s3.get_object(
                        Bucket=SKILLS_BUCKET,
                        Key=f"jobs/{job_id}/status.json"
                    )
                    status_data = json.loads(response['Body'].read().decode('utf-8'))
                    
                    return {
                        'statusCode': 200,
                        'headers': CORS_HEADERS,
                        'body': json.dumps(status_data)
                    }
                except s3.exceptions.NoSuchKey:
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
            
            s3.put_object(
                Bucket=SKILLS_BUCKET,
                Key=f"jobs/{job_id}/status.json",
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