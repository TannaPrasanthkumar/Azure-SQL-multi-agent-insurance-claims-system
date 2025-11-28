# Human Review Tab Blank Issue - Root Cause Analysis

## Problem Statement
After fraud detection completes and `fraud_claim_for_review` is set in session state, when user switches to Human Review tab, the tab renders blank (no content at all).

## Evidence from Terminal Logs

### BEFORE Processing
```
🔍 DEBUG: main() called at 21:12:08
🔍 DEBUG: fraud_claim_for_review exists: False
🔍 DEBUG: Entering tab2 block at 21:12:08
```
**Result:** Tab renders with "No fraud claim pending review" message

### DURING Processing (Fraud Detection)
```
🔍 DEBUG: Setting fraud_claim_for_review in session state...
🔍 DEBUG: fraud_claim_for_review SET! Keys: ['policy_number', 'decision', ...]
🔍 DEBUG: Session state now has 106 keys total
```
**Result:** Data successfully stored

### AFTER Processing (Expected)
User switches to Human Review tab → Script should rerun → Should see:
```
🔍 DEBUG: main() called at [time]
🔍 DEBUG: fraud_claim_for_review exists: True
🔍 DEBUG: Entering tab2 block at [time]
```

### AFTER Processing (Actual - MISSING)
**NO DEBUG OUTPUT AFTER TAB SWITCH!**

## Hypothesis 1: Script Not Rerunning After Tab Switch
**If this is true:** Streamlit is not detecting tab change as a rerun trigger

**Test:** Add debug at very top of main() before any conditional logic

## Hypothesis 2: Exception Occurring Before Tab2 Renders
**If this is true:** Some code between main() start and tab2 block is failing

**Test:** Wrap entire main() in try/except and log errors

## Hypothesis 3: Tab1 Code Blocking Tab2 Rendering
**If this is true:** After fraud detection, when script reruns, tab1's conditional logic prevents proper tab switching

**Current Structure:**
```python
with tab1:
    if fraud_detected and not tab1_reset:
        # Show "Processing complete" message
        # Button to process another claim
    else:
        # Show file uploader
        if uploaded_file:
            if st.button("Start Processing"):
                # MASSIVE WORKFLOW CODE (1500+ lines)
```

**Problem:** When `uploaded_file` still exists after fraud detection, the entire `if uploaded_file:` block runs again, potentially interfering with tab rendering.

## Hypothesis 4: Session State Being Cleared
**If this is true:** Something is deleting fraud_claim_for_review between fraud detection and tab switch

**Evidence:** Terminal shows 106 keys after setting fraud data, but no subsequent logs show the data

## Next Steps

1. **Immediate Test:** Check if ANY debug output appears when switching to Human Review tab after fraud
2. **Add Safety Logging:** Put debug statements at EVERY critical point
3. **Trace Execution Flow:** Understand exact order of operations during tab switch
4. **Isolate Tab2:** Move tab2 block to run FIRST to see if it's a rendering order issue
