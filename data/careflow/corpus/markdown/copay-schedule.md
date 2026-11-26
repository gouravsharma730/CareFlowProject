# Co-payment and Cost-Share Schedule

# MERIDIAN HEALTH PARTNERS
## OPERATIONAL POLICY MANUAL
### SECTION 4: FINANCIAL ADMINISTRATION & MEMBER COST-SHARING
#### POLICY 4.12: BENEFIT PLAN COST-SHARE SCHEDULE AND CLAIMS CALCULATION RULES
**Effective Date:** January 1, 2026  
**Issuing Authority:** Office of the Chief Financial Officer & Operations Policy Committee  
**Applicability:** All Meridian Health Partners Outpatient Clinics (Sites 01 through 14) and Revenue Cycle Systems  

---

### 4.12.1 PURPOSE AND SCOPE
This policy establishes the definitive cost-share parameters, co-pay schedules, and calculation algorithms administered by the Meridian Health Partners claims processing system for all four internal health benefit plans. Front-desk intake staff, billing specialists, and automated adjudication engines must adhere strictly to the figures and formulas detailed herein.

---

### 4.12.2 PLAN COST-SHARE SCHEDULE
The claims system shall enforce the following annual deductibles, coinsurance rates, out-of-pocket maximums, specialist co-pays, telehealth co-pays, imaging co-pays, and lab co-pays for each respective plan:

1. **MERIDIAN-GOLD**
   * Annual Deductible: $500.00
   * Coinsurance Rate: 20%
   * Out-of-Pocket Maximum: $25,000.00
   * Specialist Visit Co-pay: $25.00
   * Telehealth Co-pay: $10.00
   * Imaging Co-pay: $60.00
   * Lab Co-pay: $15.00

2. **MERIDIAN-SILVER**
   * Annual Deductible: $1,500.00
   * Coinsurance Rate: 30%
   * Out-of-Pocket Maximum: $15,000.00
   * Specialist Visit Co-pay: $45.00
   * Telehealth Co-pay: $18.00
   * Imaging Co-pay: $110.00
   * Lab Co-pay: $25.00

3. **MERIDIAN-BRONZE**
   * Annual Deductible: $3,000.00
   * Coinsurance Rate: 40%
   * Out-of-Pocket Maximum: $8,000.00
   * Specialist Visit Co-pay: $45.00
   * Telehealth Co-pay: $18.00
   * Imaging Co-pay: $160.00
   * Lab Co-pay: $35.00

4. **CIVIC-BASE**
   * Annual Deductible: $2,000.00
   * Coinsurance Rate: 35%
   * Out-of-Pocket Maximum: $10,000.00
   * Specialist Visit Co-pay: $45.00
   * Telehealth Co-pay: $18.00
   * Imaging Co-pay: $130.00
   * Lab Co-pay: $30.00

---

### 4.12.3 TELEHEALTH CO-PAY DERIVATION RULE
1. Telehealth visits shall carry a fixed co-pay equal to exactly 40 percent of the corresponding plan's in-person specialist co-pay.
2. *Verification Table:*
   * MERIDIAN-GOLD: $25.00 specialist $\times$ 0.40 = $10.00 telehealth co-pay.
   * MERIDIAN-SILVER: $45.00 specialist $\times$ 0.40 = $18.00 telehealth co-pay.
   * MERIDIAN-BRONZE: $45.00 specialist $\times$ 0.40 = $18.00 telehealth co-pay.
   * CIVIC-BASE: $45.00 specialist $\times$ 0.40 = $18.00 telehealth co-pay.

---

### 4.12.4 CLAIMS SYSTEM CALCULATION RULES
The automated co-pay and claims calculation engine shall execute the following conditional logic for all submitted encounter claims:

1. **Flat Co-pay Services:** A standard consultation, diagnostic imaging service, or laboratory service costs the patient the flat co-pay amount specified in Section 4.12.2 only. Deductible and coinsurance rules do not apply to these discrete service categories.
2. **Procedure Services:** A surgical or clinical procedure costs the patient the unmet part of the annual deductible plus the plan's specified coinsurance rate applied to the remainder of the list price.
3. **Out-of-Pocket Cap:** Patient liability calculated under Section 4.12.4(2) shall never exceed the plan's designated out-of-pocket maximum for the calendar year.
4. **Deductible Status Threshold:** Any recorded deductible-met amount in the claims database that exceeds the plan's stated annual deductible shall be automatically re-indexed by the system to mean the deductible is fully met ($0.00 unmet).

---

### 4.12.5 WORKED CALCULATION EXAMPLE
To ensure absolute audit compliance across all fourteen Meridian sites, claims examiners shall reference the following standard calculation for partially met deductibles:

* **Patient Plan:** MERIDIAN-BRONZE
* **Annual Deductible:** $3,000.00
* **Deductible Met to Date:** $1,800.00
* **Unmet Deductible:** $1,200.00 ($3,000.00 - $1,800.00)
* **Encounter Type:** Procedure — Knee Arthroscopy (Billing Code: ORTH-ARTH)
* **List Price:** $6,200.00
* **Plan Coinsurance Rate:** 40%

**Calculation Steps:**
1. Subtract the unmet deductible from the total list price:  
   $$\$6,200.00 - \$1,200.00 = \$5,000.00$$
2. Apply the MERIDIAN-BRONZE coinsurance rate (40%) to the remaining balance ($5,000.00):  
   $$\$5,000.00 \times 0.40 = \$2,000.00$$
3. Sum the unmet deductible and the coinsurance liability to determine total patient share:  
   $$\$1,200.00 \text{ (unmet deductible)} + \$2,000.00 \text{ (coinsurance)} = \$3,200.00$$
4. **Final Patient Responsibility:** The patient owes **$3,200.00** for the ORTH-ARTH procedure.
