# Human Review System - How It Works

## **Current Issues Identified:**

### **Issue 1: Workflow HTML Shown After Completion**
When ML model predicts NO fraud (legitimate claim), the workflow visualization HTML remains visible on screen showing:
- Orchestrator ✅
- Document ✅  
- Databricks ✅
- Eligibility ✅

**Expected:** After successful processing, hide the workflow progress and show clean results only.

### **Issue 2: "Live Workflow Progress" Always Displayed**
The workflow tracker is shown during processing, which clutters the UI.

**Expected:** Option to show/hide workflow progress, or remove it entirely for cleaner UX.

---

## **Human Review System Flow:**

### **Normal Flow (No Fraud):**
```
Upload PDF → Document Analysis → Databricks Validation → Eligibility Check 
→ Fraud Detection (ML: fraud_probability < 70%) → APPROVED 
→ Show success message → Auto-approve & send communication
```

### **Fraud Detected Flow:**
```
Upload PDF → Document Analysis → Databricks Validation → Eligibility Check 
→ Fraud Detection (ML: fraud_probability >= 70%) → FRAUD ALERT
→ Create Human Review Record → Save to review_queue.json 
→ Show fraud alert with metrics → Stop workflow (st.stop())
→ User switches to "Human Review" tab
```

### **Human Review Tab:**
```
User opens tab → Loads review_queue.json → Shows 8 pending reviews
→ User selects review → Sees claim details + fraud indicators
→ User fills form:
   - Reviewer Name
   - Decision: APPROVE or REJECT
   - Review Notes (minimum 20 chars)
→ Clicks Submit → Updates review_queue.json
→ Archives to review_history.json → Logs to Audit Agent
→ Shows success message with balloons
```

---

## **Current Problems:**

### **Problem 1: App Stops Immediately on Load**
- Terminal shows: "Stopping..." right after "Local URL: http://localhost:8501"
- App exits with code 1 before showing UI
- Likely causes:
  - Cached session state from previous run
  - `st.stop()` being called during page initialization
  - Databricks agent initialization failing
  
### **Problem 2: Human Review Tab Empty**
- 8 pending reviews exist in review_queue.json (verified via PowerShell)
- Tab should show all 8 reviews with APPROVE/REJECT forms
- Currently shows nothing because app stops before loading

### **Problem 3: Workflow HTML Clutter**
- Workflow visualization remains visible after processing completes
- Makes results hard to read
- Should be hidden or removed after completion

---

## **Solutions Needed:**

### **1. Fix Premature Stop Issue**
- Clear stale session state on app startup
- Wrap initialization code in try-except
- Remove or guard `st.stop()` calls that trigger too early

### **2. Fix Human Review Tab Display**
- Ensure `render_human_review_ui()` loads independently
- Don't require workflow processing to view pending reviews
- Tab should work even on fresh page load

### **3. Clean Up Workflow Display**
- Hide workflow progress after completion
- Show only final results and decision
- Keep UI clean and professional

### **4. Make Workflow Persistent**
- After review decision, claim should auto-complete remaining steps
- If APPROVED: Continue to Communication agent
- If REJECTED: Show rejection confirmation
- Don't require user to re-upload PDF

---

## **Human Review Tab Features (Already Implemented):**

### **📋 Pending Reviews Tab:**
- Lists all pending reviews from queue
- Shows review cards with:
  - Review ID, timestamp, confidence score
  - Claim info: policy number, amount, date, reason
  - Fraud indicators: driver rating, age, accident area
  - AI analysis: fraud probability, risk level
- Review form with:
  - Reviewer name input
  - APPROVE/REJECT radio buttons
  - Notes text area (required)
  - Submit button

### **📊 Review Statistics Tab:**
- Metrics: total reviews, pending, approval rate, avg confidence
- Charts: status distribution, review outcomes

### **📜 Review History Tab:**
- Table of all completed reviews
- Detailed view with reviewer notes and decisions

---

## **Quick Fix Plan:**

1. **Remove workflow visualization HTML** - Replace with simple status text
2. **Fix app startup** - Add proper session state cleanup
3. **Test Human Review tab** - Verify 8 pending reviews display correctly
4. **Add workflow continuation** - After review decision, complete remaining steps automatically
