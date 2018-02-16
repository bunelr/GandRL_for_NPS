import json
import re
import copy

class AstParseException(BaseException):
    def __init__(self, error_message):
        self.error_message = error_message

    def __str__(self):
        return self.error_message

class AstParser(object):
    def __init__(self):
        actions = [
            'move',
            'turnLeft',
            'turnRight',
            'pickMarker',
            'putMarker',
        ]

        self.action_hash = {}
        for x in actions:
            self.action_hash[x] = 1

        conditionals = [
            'markersPresent',
            'noMarkersPresent',
            'leftIsClear',
            'rightIsClear',
            'frontIsClear',
        ]

        self.conditional_hash = {}
        for x in conditionals:
            self.conditional_hash[x] = 1

    def parse(self, t):
        t = copy.deepcopy(t)
        if len(t) > 1 and t[-1] == '</s>':
            t.pop()
        self.__assert_token(t, 0, 'DEF')
        self.__assert_token(t, 1, 'run')
        self.__assert_token(t, 2, 'm(')

        result = self.__parse_block(t, 3, 'm)')
        if result['end'] != len(t)-1:
            self.__raise_exception(t, result['end'], "Unhandled tokens at end")

        obj = {}
        obj['run'] = result['block']
        return obj

    def __parse_block(self, t, start, final_delim):
        block = []
        index = start
        while True:
            if index >= len(t):
                self.__raise_exception(t, index, "Missing final delimiter '{0}'".format(final_delim))
            if t[index] == final_delim:
                break
            command_res = self.__parse_command(t, index)
            block.append(command_res['object'])
            index = command_res['end']+1
        if len(block) == 0:
            self.__raise_exception(t, index, "Blocks cannot be empty")
        result = {
            'block': block,
            'end': index
        }
        return result

    def __parse_command(self, t, index):
        obj = {}
        end = None
        command = self.__get_token(t, index)
        if command == 'REPEAT':
            obj['type'] = 'repeat'
            num_times_string = self.__get_token(t, index+1)
            m = re.match(r"R=(\d+)", num_times_string)
            if m is None:
                self.__raise_exception(t, index+1, "Not a repeat token: {0}".format(num_times_string))
            num_times = int(m.group(1))
            obj['times'] = num_times

            # body
            body_start = index+2
            self.__assert_token(t, body_start, 'r(')
            body_res = self.__parse_block(t, body_start+1, 'r)')
            obj['body'] = body_res['block']
            end = body_res['end']
        elif command == 'WHILE':
            obj['type'] = 'while'

            # cond
            cond_res = self.__parse_cond(t, index+1)
            obj['condition'] = cond_res['object']

            # body
            body_start = cond_res['end']+1
            self.__assert_token(t, body_start, 'w(')
            body_res = self.__parse_block(t, body_start+1, 'w)')
            obj['body'] = body_res['block']
            end = body_res['end']
        elif command == 'IF':
            obj['type'] = 'if'

            # cond
            cond_res = self.__parse_cond(t, index+1)
            obj['condition'] = cond_res['object']

            # body
            body_start = cond_res['end']+1
            self.__assert_token(t, body_start, 'i(')
            body_res = self.__parse_block(t, body_start+1, 'i)')
            obj['body'] = body_res['block']
            end = body_res['end']
        elif command == 'IFELSE':
            obj['type'] = 'ifElse'

            # cond
            cond_res = self.__parse_cond(t, index+1)
            obj['condition'] = cond_res['object']

            # then
            body_start = cond_res['end']+1
            self.__assert_token(t, body_start, 'i(')
            then_res = self.__parse_block(t, body_start+1, 'i)')
            obj['ifBody'] = then_res['block']

            # else
            else_start = then_res['end']+1
            self.__assert_token(t, else_start, 'ELSE')
            self.__assert_token(t, else_start+1, 'e(')
            else_res = self.__parse_block(t, else_start+2, 'e)')
            obj['elseBody'] = else_res['block']
            end = else_res['end']
        elif command in self.action_hash:
            obj['type'] = command
            end = index
        else:
            self.__raise_exception(t, index, "Unknown command: {0}".format(command))

        if end is None:
            raise Exception("'end' should never be none")

        result = {
            'object': obj,
            'end': end
        }
        return result

    def __parse_cond(self, t, index):
        obj = {}
        self.__assert_token(t, index, 'c(')
        token = self.__get_token(t, index+1)
        end = None
        if token == 'not':
            obj['type'] = 'not'
            child_res = self.__parse_cond(t, index+2)
            obj['condition'] = child_res['object']
            end = child_res['end']
        elif token in self.conditional_hash:
            obj['type'] = token
            end = index+1
        else:
            self.__raise_exception(t, index, "Unexpected conditional token: {0}".format(token))

        end += 1
        self.__assert_token(t, end, 'c)')

        result = {
            'object': obj,
            'end': end
        }
        return result

    def __get_token(self, t, index):
        if index < 0 or index >= len(t):
            self.__raise_exception(t, index, "Index out of range: {0} (length = {1})".format(index, len(t)))
        return t[index]

    def __assert_token(self, t, index, expected):
        token = self.__get_token(t, index)
        if token != expected:
            self.__raise_exception(t, index, "Unexpected token: {0}".format(expected))

    def __raise_exception(self, t, index, error_message):
        msg = "Error parsing tokens: '{0}'. Error message: {1}".format(" ".join(t), error_message)
        raise AstParseException(msg)

class AstConverter(object):
    def __init__(self):
        self.crash_protection = False

    def get_vocab_tokens(self):
        tokens = [
            # Logic
            'not',
            'and',
            'or',

            # Methods
            'DEF',
            'run',

            # Commands
            'REPEAT',
            'WHILE',
            'IF',
            'IFELSE',
            'ELSE',

            # Tests
            'markersPresent',
            'noMarkersPresent',
            'leftIsClear',
            'rightIsClear',
            'frontIsClear',

            # Actions
            'move',
            'turnLeft',
            'turnRight',
            'pickMarker',
            'putMarker',
        ]

        for x in ['m', 'c', 'r', 'w', 'i', 'e']:
            tokens.append(x+'(')
            tokens.append(x+')')

        for i in xrange(20):
            tokens.append('R={0}'.format(i))

        return tokens

    def to_tokens(self, ast):
        ast_json = ast.getJson()

        method_names = sorted(ast_json.keys())
        if 'run' in method_names:
            method_names.remove('run')
            method_names = ['run']+method_names

        tokens = []
        for method_name in method_names:
            method_json = ast_json[method_name]
            self.__make_method(method_name, method_json, tokens)
        return tokens

    def __make_method(self, name, json, tokens):
        tokens.append('DEF')
        tokens.append(name)
        tokens.append('m(')
        self.__expand_code_block(1, json, tokens)
        tokens.append('m)')

    def __expand_code_block(self, indent, code_block, tokens):
        for block in code_block:
            block_type = block['type']

            # Basic commands
            if self.__is_command(block_type):
                tokens.append(block_type)
            # For loops
            elif block_type == 'repeat':
                num_times = block['times']
                body = block['body']

                tokens.append('REPEAT')
                tokens.append('R={0}'.format(num_times))
                tokens.append('r(')
                self.__expand_code_block(indent+1, body, tokens)
                tokens.append('r)')
            # While loops
            elif block_type == 'while':
                body = block['body']
                tokens.append('WHILE')
                self.__expand_condition_block(block['condition'], tokens)
                tokens.append('w(')
                self.__expand_code_block(indent+1, body, tokens)
                tokens.append('w)')
            # If statements
            elif block_type == 'if':

                body = block['body']

                tokens.append('IF')
                self.__expand_condition_block(block['condition'], tokens)
                tokens.append('i(')
                self.__expand_code_block(indent+1, body, tokens)
                tokens.append('i)')

            # If/else statements
            elif block_type == 'ifElse':
                if_body = block['ifBody']
                else_body = block['elseBody']

                tokens.append('IFELSE')
                self.__expand_condition_block(block['condition'], tokens)
                tokens.append('i(')
                self.__expand_code_block(indent+1, if_body, tokens)
                tokens.append('i)')
                tokens.append('ELSE')
                tokens.append('e(')
                self.__expand_code_block(indent+1, else_body, tokens)
                tokens.append('e)')

            # Invoking user defined methods
            elif block_type == 'invoke':
                raise Exception('Multiple methods not supported')
                # methodName = block['method']
                # codestr += self.getIndent(indent)+methodName+'()\n'

            # Opps! There must have been a parse error.
            else:
                raise Exception('unknown type: \''+block_type+'\'')

    def __expand_condition_block(self, block, tokens):
        if not 'type' in block:
            raise Exception('block has no type: \''+block+'\'')
        tokens.append('c(')
        block_type = block['type']
        if self.__is_condition_test(block_type):
            tokens.append(block_type)
        elif block_type == 'not':
            tokens.append('not')
            conditionStr = self.__expand_condition_block(block['condition'], tokens)
        else:
            raise Exception('unknown type: \''+block_type+'\'')
        tokens.append('c)')

    def __is_condition_test(self, block_type):
        if block_type == 'markersPresent': return True
        if block_type == 'noMarkersPresent': return True
        if block_type == 'leftIsClear': return True
        if block_type == 'rightIsClear': return True
        if block_type == 'frontIsClear': return True
        return False

    def __is_command(self, block_type):
        if block_type == 'move': return True
        if block_type == 'turnLeft': return True
        if block_type == 'turnRight': return True
        if block_type == 'pickMarker': return True
        if block_type == 'putMarker': return True
        return False
