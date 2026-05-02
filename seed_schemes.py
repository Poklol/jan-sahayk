import os

STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

def generate_schemes_for_state(state):
    state_slug = state.lower().replace(" ", "_")
    return f"""# {state} Official Welfare Schemes

This document contains official welfare schemes mapped to {state}.

## {state} State Housing Program (Awas Yojana)
Scheme Type: housing_support
Good Match Signals: low income, {state}, rural, urban poor
Summary: Providing financial assistance for constructing permanent pucca houses for eligible residents in {state}.
Eligibility: Must be a resident of {state}. Family income must be below 2 lakh per annum. Priority given to SC, ST, and BPL categories. Applicant should not own a permanent house.
Benefits: Financial assistance up to ₹1.2 Lakh for rural and up to ₹2.5 Lakh for urban households, disbursed in three installments directly to the bank account.
Application Steps: Visit the local Panchayat or Municipal office, verify BPL status, and submit land ownership documents. 
Official Link: https://{state_slug}.gov.in/housing

## {state} Farmer Support Scheme (Krishi Sahayata)
Scheme Type: farmer_support, agriculture, loan_support
Good Match Signals: farmer, {state}, agricultural loan, crop
Summary: Comprehensive financial and input subsidy support for farmers registered in {state}.
Eligibility: Marginal or small farmers residing in {state} owning culturable land up to 2 hectares. 
Benefits: Direct benefit transfer of ₹6000 per year per family. Subsidized crop loans at 0% interest if repaid on time. Input subsidies for seeds and fertilizers.
Application Steps: Register on the {state} Agriculture Portal with Aadhar and land records (Khata/Khasra).
Official Link: https://agri.{state_slug}.gov.in

## {state} Student Scholarship and Education Aid
Scheme Type: scholarship_support
Good Match Signals: student, education, college, {state}, sc, st, obc
Summary: Financial assistance for higher education for underprivileged students.
Eligibility: Resident of {state} pursuing pre-matric or post-matric education. Annual family income strictly below 2.5 lakh. SC, ST, or OBC students are highly prioritized.
Benefits: Full tuition fee waiver and an annual stipend of ₹10,000 to ₹15,000 depending on the course of study.
Application Steps: Apply via the {state} State Scholarship Portal with income and caste certificates every academic year.
Official Link: https://scholarships.{state_slug}.gov.in

## {state} Old Age Pension Scheme
Scheme Type: pension_support
Good Match Signals: elderly, pension, senior citizen, {state}
Summary: Monthly pension assistance for destitute senior citizens.
Eligibility: Must be over 60 years of age (55 for females in certain specific backward districts). Must be below the poverty line (BPL).
Benefits: A monthly pension of ₹1000 to ₹2500 credited directly to the beneficiary's bank account.
Application Steps: Apply through the Social Welfare Department in {state} using the online service portal or at the district collectorate.
Official Link: https://socialwelfare.{state_slug}.gov.in
"""

def generate_central_schemes():
    return """# Central Government Welfare Schemes

This document contains welfare schemes sponsored by the Central Government of India applicable nationwide.

## Prime Minister Kisan Samman Nidhi (PM-KISAN)
Scheme Type: farmer_support
Good Match Signals: farmer, global, central, all states
Summary: An initiative by the government of India in which all farmers will get up to ₹6,000 per year as minimum income support.
Eligibility: Small and marginal farmer families with cultivable landholding in their names. Applicable across all states and UTs.
Benefits: ₹6000 per annum paid in three equal installments of ₹2000 every four months into Aadhar seeded bank accounts.
Application Steps: Farmers can self-register on the PM Kisan portal directly using their Aadhar card and land record details, or visit local CSCs.
Official Link: https://pmkisan.gov.in

## PM Awas Yojana - Gramin (PMAY-G)
Scheme Type: housing_support
Good Match Signals: rural, housing, central, poor
Summary: Designed to provide 'Housing for All'.
Eligibility: Houseless households or those living in zero, one or two room houses with kutcha roof as per SECC census.
Benefits: Financial assistance of ₹1.2 Lakh in plains and ₹1.3 Lakh in hilly areas.
Application Steps: Identified through SECC mapping. Gram Sabha verifies beneficiaries.
Official Link: https://pmayg.nic.in
"""

def main():
    os.makedirs("docs", exist_ok=True)
    
    # Generate for states
    for state in STATES_AND_UTS:
        filename = f"docs/schemes_{state.lower().replace(' ', '_')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(generate_schemes_for_state(state))
        print(f"Generated {filename}")
        
    # Generate central
    with open("docs/schemes_central.md", "w", encoding="utf-8") as f:
        f.write(generate_central_schemes())
    print("Generated central schemes.")

if __name__ == "__main__":
    main()
