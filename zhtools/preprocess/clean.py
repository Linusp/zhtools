import unicodedata


def to_halfwidth(text):
    result = ''
    for char in text:
        name = unicodedata.name(char, None)
        if name == 'IDEOGRAPHIC SPACE':
            result += ' '
        elif not name or name.find('FULLWIDTH') != 0:
            result += char
        else:
            new_name = name.replace('FULLWIDTH', '').strip()
            new_char = unicodedata.lookup(new_name)
            result += new_char

    return result
