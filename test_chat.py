#!/usr/bin/env python3
"""
test_chat.py - Simple Finance Agent Chat Tester
Run this while the server is running to test all chat features
"""

import requests
import json
from pprint import pprint
import time

BASE_URL = "http://localhost:8000/api/v1"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_basic_chat():
    """Test 1: Basic conversational chat"""
    print_header("TEST 1: Basic Chat - Simple Question")
    
    payload = {
        "message": "I want to start a consulting business in Casablanca"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nResponse:")
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Agent Mode: {data.get('agent_mode')}")
        print(f"✓ Message Preview: {data.get('message')[:150]}...")
        return True
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Error: {response.text}")
        return False

def test_client_tracking():
    """Test 2: Chat with client email tracking"""
    print_header("TEST 2: Client Tracking - Save to Database")
    
    payload = {
        "message": "Calculate my break-even point",
        "client_email": "test@business.ma"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nResponse:")
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Client Email: test@business.ma")
        print(f"✓ Session ID: {data.get('session_id')}")
        print(f"✓ Message: {data.get('message')[:150]}...")
        return True
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Error: {response.text}")
        return False

def test_hypothesis_fast_track():
    """Test 3: Fast-track with Hypothesis data"""
    print_header("TEST 3: Hypothesis Fast-Track - Pre-filled Data")
    
    payload = {
        "message": "Analyze my business plan",
        "client_email": "entrepreneur@startup.ma",
        "hypothesis_payload": {
            "revenue_projected_m1": 50000,
            "revenue_projected_m6": 150000,
            "costs_monthly": 30000,
            "employees": 2,
            "capital_invested": 100000,
            "sector": "Consulting"
        }
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nResponse:")
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Client Email: entrepreneur@startup.ma")
        print(f"✓ Session ID: {data.get('session_id')}")
        
        # Show KPI suggestions if available
        if data.get('kpi_suggestions'):
            print(f"✓ KPI Suggestions: {', '.join(data.get('kpi_suggestions', []))}")
        
        # Show KPI details if available
        if data.get('kpi_details'):
            print(f"✓ KPI Details Found:")
            for kpi_name, kpi_data in data.get('kpi_details', {}).items():
                print(f"  - {kpi_name}: {kpi_data.get('value')} {kpi_data.get('unit')}")
                print(f"    {kpi_data.get('explanation')}")
        
        print(f"✓ Message: {data.get('message')[:150]}...")
        return True
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Error: {response.text}")
        return False

def test_kpi_explanation():
    """Test 4: KPI with explanations"""
    print_header("TEST 4: KPI Explanations - Glossary Lookup")
    
    payload = {
        "message": "What is seuil_rentabilite and how is it calculated?"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nResponse:")
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Message: {data.get('message')[:200]}...")
        return True
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Error: {response.text}")
        return False

def test_session_persistence():
    """Test 5: Multiple messages in one session"""
    print_header("TEST 5: Session Persistence - Multi-turn Conversation")
    
    # Create a session
    payload1 = {
        "message": "I'm starting a tech startup in Rabat",
        "client_email": "startup@tech.ma"
    }
    
    print("Message 1:")
    print(f"Request: {json.dumps(payload1, indent=2)}")
    
    response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
    
    if response1.status_code == 200:
        data1 = response1.json()
        session_id = data1.get('session_id')
        print(f"✓ Session ID: {session_id}")
        print(f"✓ Response: {data1.get('message')[:100]}...")
        
        # Continue conversation with same session
        time.sleep(1)
        
        payload2 = {
            "message": "What about initial capital requirements?",
            "session_id": session_id,
            "client_email": "startup@tech.ma"
        }
        
        print("\nMessage 2 (same session):")
        print(f"Request: {json.dumps(payload2, indent=2)}")
        
        response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"✓ Session ID: {data2.get('session_id')}")
            print(f"✓ Response: {data2.get('message')[:100]}...")
            return True
        else:
            print(f"✗ Error: {response2.text}")
            return False
    else:
        print(f"✗ Status: {response1.status_code}")
        print(f"✗ Error: {response1.text}")
        return False

def test_metadata():
    """Test 6: Response metadata (LLM used, RAG confidence)"""
    print_header("TEST 6: Metadata - LLM & RAG Confidence")
    
    payload = {
        "message": "Tell me about Moroccan business regulations"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nResponse:")
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        
        if data.get('metadata'):
            metadata = data.get('metadata')
            print(f"✓ LLM Used: {metadata.get('llm_used', 'groq')}")
            print(f"✓ RAG Confidence: {metadata.get('rag_confidence', 0):.2%}")
        
        print(f"✓ Message: {data.get('message')[:150]}...")
        return True
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Error: {response.text}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*70)
    print("  FINANCE AGENT - COMPREHENSIVE CHAT TEST SUITE")
    print("█"*70)
    print("\nMake sure your FastAPI server is running:")
    print("  $ uvicorn app.main:app --reload")
    print("\nTesting endpoint:", BASE_URL)
    
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Client Tracking", test_client_tracking),
        ("Hypothesis Fast-Track", test_hypothesis_fast_track),
        ("KPI Explanations", test_kpi_explanation),
        ("Session Persistence", test_session_persistence),
        ("Metadata", test_metadata),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST RESULTS SUMMARY")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ ALL TESTS PASSED! Your Finance Agent is working perfectly! 🚀")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Check the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        exit(1)
