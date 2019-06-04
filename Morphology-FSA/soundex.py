'''
Daniel Vilinsky
'''

from fst import FST
import string, sys
from fsmutils import compose


def letters_to_numbers():
    """
    Returns an FST that converts letters to numbers as specified by
    the soundex algorithm
    """
    #State variables:
    start_state = 'start'
    gobbled = 'gobbled'

    states = {
        'labials': {'b', 'f', 'p', 'v', 'B', 'F', 'P', 'V'},
        #not an entirely accurate name
        'alveolars': {'c', 'g', 'j', 'k', 'q', 's', 'x', 'z', 'C', 'G', 'J', 'K', 'Q',
                      'S', 'X', 'Z'},
        'dentals': {'d', 't', 'D', 'T'},
        'lateral': {'l', 'L'},
        'nasals': {'m', 'n', 'M', 'N'},
        'rhotic': {'r', 'R'},
        'vowels': {'a', 'e', 'h', 'i', 'o', 'u', 'w', 'y', 'A', 'E', 'H', 'I', 'O', 'U',
                   'W', 'Y'}
    }
    state_output = {'labials': '1', 'alveolars': '2', 'dentals': '3', 'lateral': '4',
                    'nasals': '5', 'rhotic': '6', 'vowels': ''}

    # Let's define our first FST
    f1 = FST('soundex-generate')

    #Inital consuming
    f1.add_state(start_state)
    f1.initial_state = start_state
    f1.add_state(gobbled)
    for letter in string.ascii_letters:
        f1.add_arc(start_state, gobbled, letter, letter)

    for key in states:
        f1.add_state(key)
        f1.set_final(key) #every letter type is a final state

    for letter_type in states:
        for letter in states[letter_type]:
            f1.add_arc(gobbled, letter_type, letter, state_output[letter_type])
            f1.add_arc(letter_type, letter_type, letter, '')

    #Transition from every state to every other state
    for letter_type in states:
        for other in set(states.keys()) - {letter_type}:
            for letter in states[other]:
                f1.add_arc(letter_type, other, letter, state_output[other])
    return f1


def truncate_to_three_digits():
    """
    Create an FST that will truncate a soundex string to three digits
    """
    start_state = 'start'
    letter_first = 'letter_first'
    number_first = 'number_first'
    numbers = list('0123456789')

    # Initialization
    f2 = FST('soundex-truncate')
    f2.add_state(start_state)
    f2.add_state(letter_first)
    f2.add_state(number_first)
    f2.set_final(number_first) #Don't think this would ever occur, but tests want it
    f2.initial_state = start_state

    for letter in string.ascii_letters:
        f2.add_arc(start_state, letter_first, letter, letter)
    for number in numbers:
        f2.add_arc(start_state, number_first, str(number), str(number))

    get_letter_number(f2, letter_first, numbers)
    get_number_letter(f2, number_first, numbers)

    return f2

# Get the 2 transitions following reading in an initial number
# for the second transducer
def get_number_letter(f2, number_first, numbers):
    previous1 = number_first
    for i in range(2):
        current1 = 'number_letter' + str(i + 1)  # namespace is getting polluted
        f2.add_state(current1)
        f2.set_final(current1)
        for num in numbers:
            f2.add_arc(previous1, current1, str(num), str(num))
        previous1 = current1
        if i == 1:
            for num in numbers:
                f2.add_arc(previous1, previous1, str(num), '')

# Get the three transitions following reading in an initial letter
# for the second transducer
def get_letter_number(f2, letter_first, numbers):
    previous = letter_first
    for i in range(3):
        current = 'letter_number' + str(i + 1)
        f2.add_state(current)
        f2.set_final(current)
        for num in numbers:
            f2.add_arc(previous, current, str(num), str(num))
        previous = current
        if i == 2:
            for num in numbers:
                f2.add_arc(previous, previous, str(num), '')

def add_zero_padding():
    # Now, the third fst - the zero-padding fst

    #Variable aliases
    start_state = 'start'
    just_numbers = 'just_numbers'
    letter_first = 'letter_first'
    epsilons = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5']

    #Initialization
    f3 = FST('soundex-padzero')
    f3.add_state(start_state)
    f3.add_state(just_numbers)
    f3.add_state(letter_first)
    f3.initial_state = start_state
    add_numbers(f3, start_state, just_numbers)
    for letter in string.ascii_letters:
        f3.add_arc(start_state, letter_first, letter, letter)

    build_letter_first(f3, epsilons, letter_first)
    build_number_first(f3, epsilons, just_numbers)

    return f3


def build_number_first(f3, epsilons, just_numbers):
    #Variable aliases
    just_numbers_one = 'just_numbers_one'
    just_numbers_two = 'just_numbers_two'
    just_numbers_three = 'just_numbers_three'

    f3.add_state(epsilons[3])
    f3.add_state(epsilons[4])
    f3.add_arc(just_numbers, epsilons[3], '', '0')
    f3.add_arc(epsilons[3], epsilons[4], '', '0')
    f3.set_final(epsilons[4])

    f3.add_state(just_numbers_one)
    f3.add_state(epsilons[5])
    add_numbers(f3, just_numbers, just_numbers_one)
    f3.add_arc(just_numbers_one, epsilons[5], '', '0')
    f3.set_final(epsilons[5])

    f3.add_state(just_numbers_two)
    f3.add_state(just_numbers_three)
    add_numbers(f3, just_numbers, just_numbers_two)
    add_numbers(f3, just_numbers_two, just_numbers_three)
    f3.set_final(just_numbers_three)

def build_letter_first(f3, epsilons, letter_first):
    #Variable aliases
    one_digit = 'one_digit'
    two_digits = 'two_digits'
    two_digits_second = 'two_digits_second'
    three_digits = 'three_digits'
    three_digits_second = 'three_digits_second'
    three_digits_third = 'three_digits_third'

    f3.add_state(one_digit)
    add_numbers(f3, letter_first, one_digit)
    # TODO: make into parameterized function
    f3.add_state(epsilons[0])
    f3.add_arc(one_digit, epsilons[0], '', '0')
    f3.add_state(epsilons[1])
    f3.add_arc(epsilons[0], epsilons[1], '', '0')
    f3.set_final(epsilons[1])

    f3.add_state(two_digits)
    f3.add_state(two_digits_second)
    # TODO: HIGHER ORDER PROCEDURES would be useful here, but w/ changing states hard to do
    add_numbers(f3, letter_first, two_digits)
    add_numbers(f3, two_digits, two_digits_second)
    f3.add_state(epsilons[2])
    f3.add_arc(two_digits_second, epsilons[2], '', '0')
    f3.set_final(epsilons[2])

    f3.add_state(three_digits)
    f3.add_state(three_digits_second)
    f3.add_state(three_digits_third)
    add_numbers(f3, letter_first, three_digits)
    add_numbers(f3, three_digits, three_digits_second)
    add_numbers(f3, three_digits_second, three_digits_third)
    f3.set_final(three_digits_third)

def add_numbers(f, start_state, end_state):
    numbers = list('1234567890')
    for num in numbers:
        f.add_arc(start_state, end_state, str(num), str(num))

def soundex_convert(name_string):
    """Combine the three FSTs above and use it to convert a name into a Soundex"""
    f1 = letters_to_numbers()
    f2 = truncate_to_three_digits()
    f3 = add_zero_padding()
    return ''.join(compose(name_string, f1, f2, f3)[0])


if __name__ == '__main__':
    # for Python 2, change input() to raw_input()
    print('type a name: ',)
    user_input = input().strip()
    if user_input:
        print("%s -> %s" % (user_input, soundex_convert(tuple(user_input))))
