'''
Daniel Vilinsky
'''

import re


def is_phone_number(input_string):
    return re.search(r'(\(\d{1,3}\)|^\d{1,3})-\d{1,3}-?\d{1,4}$', input_string) is not None


def is_email_address(input_string):
    return re.search(r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]', input_string) is not None
