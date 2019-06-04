'''
Daniel Vilinsky
'''

from fst import FST


class Parser():
    def __init__(self):
        pass

    def generate(self, analysis):
        """Generate the morphologically correct word

        e.g.
        p = Parser()
        analysis = ['p','a','n','i','c','+past form']
        p.generate(analysis)
        ---> 'panicked'
        """
        start_state = 'start'

        f = FST('generator')
        f.add_state(start_state)
        f.initial_state = start_state
        self._build_generator_fst(f, analysis, start_state)

        return ''.join(f.transduce(analysis)[0])

    def _build_generator_fst(self, f, analysis, start):
        previous_state = start
        final_state = 'final_state'
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for char in analysis[:len(analysis)-1]:
            f.add_state(char)
            #Add arc from current state, to the next state (a char) and spit
            #the char back out
            f.add_arc(previous_state, char, char, char)
            previous_state = char
        f.add_state(final_state)
        f.set_final(final_state)
        if analysis[-3] in vowels:
            self._add_ending(f, analysis, previous_state, final_state, 'k')
        else:
            self._add_ending(f, analysis, previous_state, final_state)

    def _add_ending(self, f, analysis, previous_state, next_state, k=''):
        if analysis[-1] == '+past form':
            f.add_arc(previous_state, next_state, '+past form', k + 'ed')
        else:
            f.add_arc(previous_state, next_state, analysis[-1], k + 'ing')

    def parse(self, word):
        """Parse a word morphologically

        e.g.
        p = Parser()
        word = ['p', 'a', 'n', 'i', 'c', 'k','e','d']
        p.parse(word)
        ---> 'panic+past form'
        """
        lexicon = {'panic', 'havoc', 'sync', 'lick', 'want'}
        start_state = 'start'
        k_insertion = 'k_insertion'


        f = FST('parser')
        f.add_state(start_state)
        f.initial_state = start_state

        #Add paths for each word
        previous = start_state
        for vocab in lexicon:
            for char in vocab:
                current = vocab + '-' + char #uniquely identify
                f.add_state(current)
                f.add_arc(previous, current, char, char)
                previous = current
            f.add_state(k_insertion + '-' + vocab)
            f.add_arc(previous, k_insertion+'-'+vocab, 'k', '')
            self._add_ending_states(f, vocab, k_insertion+'-'+vocab, k=k_insertion)
            self._add_ending_states(f, vocab, previous)
            previous = start_state

        return ''.join(f.transduce(word)[0])

    def _add_ending_states(self, f, word, previous_state, k=''):
        #Variable aliases
        ed_state  = 'ed-' + word + k
        ing_state = 'ing-' + word + k
        d_state = 'd_state' + word + k
        n_state = 'n_state' + word + k
        g_state = 'g_state' + word + k

        f.add_state(ed_state)
        f.add_state(ing_state)


        f.add_arc(previous_state, ed_state, 'e', '+past form')
        f.add_state(d_state)
        f.set_final(d_state)
        f.add_arc(ed_state, d_state, 'd', '')

        f.add_arc(previous_state, ing_state, 'i', '+present participle')
        f.add_state(n_state)
        f.add_state(g_state)
        f.set_final(g_state)
        f.add_arc(ing_state, n_state, 'n', '')
        f.add_arc(n_state, g_state, 'g', '')
