#!/usr/bin/env python3
"""
Test script for async knowledge extraction API
"""

import requests
import base64
import json
import time

# Your API endpoint
API_URL = "https://upx0krk583.execute-api.us-west-2.amazonaws.com/prod"

def submit_job(filepath: str, enterprise_name: str = None):
    """Submit extraction job"""
    
    # Read and encode file
    with open(filepath, 'rb') as f:
        file_content = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        'file': file_content,
        'filename': filepath.split('\\')[-1].split('/')[-1],
    }
    
    if enterprise_name:
        payload['enterprise_name'] = enterprise_name
    
    print("📤 Submitting extraction job...")
    print(f"   File: {filepath}")
    print(f"   Enterprise: {enterprise_name or 'Auto-detect'}")
    print()
    
    response = requests.post(
        f"{API_URL}/extract",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 202:
        result = response.json()
        print("✅ Job submitted successfully!")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Estimated time: {result['estimated_completion']}")
        print()
        return result['job_id']
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)
        return None


def check_status(job_id: str):
    """Check job status"""
    
    response = requests.get(
        f"{API_URL}/status/{job_id}",
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)
        return None


def wait_for_completion(job_id: str, check_interval: int = 30):
    """Poll job status until complete"""
    
    print("⏳ Waiting for extraction to complete...")
    print(f"   Checking every {check_interval} seconds")
    print()
    
    while True:
        status = check_status(job_id)
        
        if not status:
            break
        
        if status['status'] == 'completed':
            print("✅ EXTRACTION COMPLETE!")
            print()
            print(json.dumps(status, indent=2))
            return status
        
        elif status['status'] == 'failed':
            print("❌ EXTRACTION FAILED!")
            print()
            print(json.dumps(status, indent=2))
            return status
        
        elif status['status'] == 'processing':
            elapsed = time.time()
            print(f"   ⏳ Still processing... (Job ID: {job_id})")
        
        time.sleep(check_interval)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_async_api.py <FILE_PATH> [ENTERPRISE_NAME]")
        print()
        print("Example:")
        print("  python test_async_api.py retail_sop_doc.md RetailCo")
        sys.exit(1)
    
    filepath = sys.argv[1]
    enterprise_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Submit job
    job_id = submit_job(filepath, enterprise_name)
    
    if job_id:
        # Wait for completion
        result = wait_for_completion(job_id)
        
        if result and result['status'] == 'completed':
            print()
            print("📦 Generated Skills:")
            for path in result.get('s3_paths', []):
                print(f"   - {path}")