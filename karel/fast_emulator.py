import copy
import sys
import os

class EmuLocationTuple(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __str__(self):
        return "{0}:{1}".format(self.name, self.index)

class EmuLocation(object):
    def __init__(self, tuples):
        self.tuples = tuples

    def add(self, name, index):
        tuples = copy.deepcopy(self.tuples)
        tuples.append(EmuLocationTuple(name, index))
        return EmuLocation(tuples)

    def __str__(self):
        return " ".join([str(x) for x in self.tuples])

class EmuTick(object):
    def __init__(self, location, type, value):
        self.location = location
        self.type = type
        self.value = value

class EmuResult(object):
    def __init__(self, status, inpgrid, outgrid, ticks, actions):
        self.status = status
        self.inpgrid = inpgrid
        self.outgrid = outgrid
        self.ticks = ticks
        self.actions = actions
        self.crashed = False

class FastEmuException(BaseException):
    def __init__(self, status):
        self.status = status

class EmuState(object):
    def __init__(self, world, max_ticks, max_actions):
        self.world = world
        self.max_ticks = max_ticks
        self.max_actions = max_actions
        self.crashed = False
        self.ticks = []
        self.actions = []

    def add_action(self, location, type):
        action_index = len(self.actions)
        self.__add_tick(EmuTick(location, 'action', action_index))
        self.actions.append(type)

    def add_condition_tick(self, location, result):
        self.__add_tick(EmuTick(location, 'condition', result))

    def add_repeat_tick(self, location, index):
        self.__add_tick(EmuTick(location, 'repeat', index))

    def __add_tick(self, tick):
        if self.max_ticks is not None and \
            self.max_ticks != -1 and \
            len(self.ticks) >= self.max_ticks:
                raise FastEmuException('MAX_TICKS')
        self.ticks.append(tick)

    def __add_action(self, action):
        if self.max_actions is not None and \
            self.max_actions != -1 and \
            len(self.actions) >= self.max_actions:
                raise FastEmuException('MAX_ACTIONS')
        self.actions.append(action)

class FastEmulator(object):
    def __init__(self, max_ticks=None, max_actions=None):
        self.max_ticks = max_ticks
        self.max_actions = max_actions
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

    def emulate(self, ast, inpgrid):
        j_ast = ast.getJson()
        world = copy.deepcopy(inpgrid)
        state = EmuState(world, self.max_ticks, self.max_actions)
        location = EmuLocation([])

        status = 'OK'
        try:
            self.__emulate_block(j_ast, 'run', location, state)
        except FastEmuException as e:
            status = e.status

        result = EmuResult(status, inpgrid, state.world, state.ticks, state.actions)

        return result

    def __emulate_condition(self, condition, location, state):
        result = self.__eval_condition_recursive(condition, state)
        state.add_condition_tick(location, result)
        return result

    def __eval_condition_recursive(self, condition, state):
        type = condition['type']
        if type == 'not':
            result = self.__eval_condition_recursive(condition['condition'], state)
            return not result
        if type not in self.conditional_hash:
            raise Exception("Type not supported: {0}".format(type))

        if type == 'noMarkersPresent':
            result = not state.world.markersPresent()
        else:
            conditional_func = getattr(state.world, type)
            result = conditional_func()
        return result

    def __emulate_block(self, parent, relationship, location, state):
        block = parent[relationship]
        for st_idx, node in enumerate(block):
            child_location = location.add(relationship, st_idx)
            type = node['type']
            if type in self.action_hash:
                action_func = getattr(state.world, type)
                action_func()
                state.add_action(child_location, type)
                if state.world.isCrashed():
                    raise FastEmuException('CRASHED')
            elif type == 'repeat':
                times = node['times']
                for i in xrange(times):
                    state.add_repeat_tick(child_location, i)
                    self.__emulate_block(node, 'body', child_location, state)
            elif type == 'while':
                while True:
                    res = self.__emulate_condition(node['condition'], child_location, state)
                    if not res:
                        break
                    self.__emulate_block(node, 'body', child_location, state)
            elif type == 'if':
                if self.__emulate_condition(node['condition'], child_location, state):
                    self.__emulate_block(node, 'body', child_location, state)
            elif type == 'ifElse':
                if self.__emulate_condition(node['condition'], child_location, state):
                    self.__emulate_block(node, 'ifBody', child_location, state)
                else:
                    self.__emulate_block(node, 'elseBody', child_location, state)
            else:
                raise Exception("Unknown type: {0}".format(type))
