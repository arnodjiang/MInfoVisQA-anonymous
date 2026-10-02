pass
import re
NUMBER = re.compile('[+\\-−]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\\.[0-9]+)?|[+\\-−]?\\.[0-9]+')

def is_plain_number(answer):
    return isinstance(answer, (str, int, float)) and (not isinstance(answer, bool)) and (NUMBER.fullmatch(str(answer).strip()) is not None)

def with_reply(question, answer, instruction):
    return question if is_plain_number(answer) else question + '\n' + instruction
