def align_tokens(tokens, text):
    point, spans = 0, []
    for token in tokens:
        start = text.find(token, point)
        if start < 0:
            raise ValueError(f'substring "{token}" not found in "{text}"')

        end = start + len(token)
        spans.append((start, end))
        point = end

    return spans
