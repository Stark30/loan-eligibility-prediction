from pydantic import BaseModel

class LoanRequest(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str  
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str
