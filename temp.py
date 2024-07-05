from faker import Faker
import random

fake = Faker()

def generate_contract():
    # Generate fake company names and addresses
    company_a_name = fake.company()
    company_b_name = fake.company()
    company_a_address = fake.address()
    company_b_address = fake.address()

    # Generate random effective date and term of agreement
    effective_date = fake.date_this_year()
    term_of_agreement = random.randint(1, 5)  # Random term between 1 and 5 years

    # Generate financial data
    initial_investment = random.randint(10000, 100000)
    profit_share_a = random.randint(30, 70)
    profit_share_b = 100 - profit_share_a

    # Generate contract text
    contract_text = f'''
    Business Partnership Agreement

    This Business Partnership Agreement ("Agreement") is entered into on {effective_date}, by and between:

    {company_a_name}, a company organized and existing under the laws of {fake.country()}, with its principal place of business located at {company_a_address} (hereinafter referred to as "Company A"), and

    {company_b_name}, a company organized and existing under the laws of {fake.country()}, with its principal place of business located at {company_b_address} (hereinafter referred to as "Company B").

    1. Purpose of Agreement

    Company A and Company B hereby agree to enter into a partnership to jointly develop and market {fake.catch_phrase()}.

    2. Roles and Responsibilities

    2.1 Company A shall be responsible for:
       - Development of the product/service.
       - Marketing strategies and implementation.
       - Providing necessary resources for the partnership.

    2.2 Company B shall be responsible for:
       - Sales and distribution of the product/service.
       - Providing market insights and customer feedback.
       - Contributing financially to the partnership as agreed upon.

    3. Term of Agreement

    This Agreement shall commence on {effective_date} and shall remain in full force and effect for a period of {term_of_agreement} years, unless terminated earlier by mutual agreement of both parties.

    4. Financial Arrangements

    4.1 Both parties agree to share the profits generated from the partnership in proportion to their respective contributions.

    4.2 Company B shall make an initial investment of ${initial_investment} into the partnership, payable within {random.randint(1, 30)} days of the Effective Date.

    5. Confidentiality

    Both parties agree to maintain the confidentiality of all proprietary information shared during the course of the partnership. This includes, but is not limited to, business plans, financial data, and customer information.

    6. Termination

    6.1 Either party may terminate this Agreement upon {random.randint(30, 180)} days' written notice to the other party.

    6.2 In the event of termination, both parties shall cooperate in winding up the affairs of the partnership in an orderly manner.

    7. Governing Law

    This Agreement shall be governed by and construed in accordance with the laws of {fake.country()}.

    8. Entire Agreement

    This Agreement constitutes the entire understanding between the parties with respect to the subject matter hereof and supersedes all prior agreements and understandings, whether written or oral, relating to such subject matter.

    IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the Effective Date first written above.

    {company_a_name}

    By: {fake.name()}, {fake.job()}

    Date: {fake.date_this_decade()}

    {company_b_name}

    By: {fake.name()}, {fake.job()}

    Date: {fake.date_this_decade()}
    '''
    return contract_text

# Example usage:
contract = generate_contract()
print(contract)
