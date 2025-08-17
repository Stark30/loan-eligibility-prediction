# These dictionaries are used to map categorical inputs to standardized values
# This is a utility module for normalization of categorical inputs in loan requests

education_map = {
    "High School": "High School",
    "Bachelor": "Bachelor's",
    "Bachelors": "Bachelor's",
    "Master": "Master's",
    "Masters": "Master's",
    "PhD": "PhD",
    "Other": "Other"
}

yes_no_map = {"yes": "Yes", "no": "No"}

employment_map = {
    "Full-time": "Full-time",
    "Full time": "Full-time",
    "Part-time": "Part-time",
    "Part time": "Part-time",
    "Unemployed": "Unemployed",
    "Other": "Other"
}

marital_map = {
    "Single": "Single",
    "Married": "Married",
    "Divorced": "Divorced",
    "Widowed": "Widowed",
    "Other": "Other"
}

loan_purpose_map = {
    "Auto": "Auto",
    "Home": "Home",
    "Education": "Education",
    "Business": "Business",
    "Other": "Other"
}
