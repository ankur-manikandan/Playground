import re

def match_emp(emp_nm, email_adrs):

    # Extract domain name
    domain_nm = email_adrs.split('@')[-1].split('.')[0]
    #print(domain_nm)
    # Extract only characters from employer name
    emp = "".join(re.findall(r'[a-zA-Z0-9]+', emp_nm.lower()))
    #print(emp)

    if domain_nm in emp:
        print(True)
    else:
        print(False)

emp_nm = 'Wal-mart pharmacy'
email_address = ['first.name@walmart.com',
                'first.name@walmartp.com',
                'first.name@walmartphmcy.com']
                
for i in email_address:
    match_emp(emp_nm, i)
    
"""
The above function will do great for email
addresses with domain names that are essentially
substrings of the employer name, example - 
Employer name: Walmart pharmacy and 
Email Address: first.last@walmartp.com.

But if we have an email address, like, first.name@walmartphmcy.com
then the above function will return False. We can have a 
secondary check using fuzzy matchin techniques. In this case,
Levenshtein distance will be a good algorithm to start with.
"""