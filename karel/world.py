import numpy as np
import torch
import math

MATRIX_DIMENSIONS = 4
MAX_API_CALLS = 1e5
MAX_MARKERS_PER_SQUARE = 101

# Class: World
# ------------
# Defines a grid world of cells with:
# - A hero which exists in a certain cells
# - Blocked cells that the hero can't pass through
# - Markers that the hero can pick up and place
# The hero can face North, South, East or West. 
# Hero only gets limited number of function calls
class World:

    # Function: Init
    # --------------
    # Creates a world from a json object. The json
    # must specify:
    # - rows and cols
    # - heroRow, heroCol and heroDir
    # - blocked cells
    # - markers.
    # See tasks/cs106a for examples
    def __init__(self, rows, cols, heroRow, heroCol, heroDir, blocked, markers):
        self.numAPICalls = 0
        self.rows = rows
        self.cols = cols
        self.heroRow = heroRow
        self.heroCol = heroCol
        self.heroDir = heroDir
        self.blocked = blocked
        self.markers = markers
        self.crashed = False

    @classmethod
    def fromJson(cls, json):
        rows = json['rows']
        cols = json['cols']
        heroRow = json['heroRow']
        heroCol = json['heroCol']
        heroDir = json['heroDir']

        blocked = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                if json['blocked'][r][c] == '*':
                    blocked[rows - r - 1][c] = 1
        markers = np.zeros((rows, cols))
        for markerJson in json['markers']:
            r = int(markerJson['r'])
            c = int(markerJson['c'])
            num = int(markerJson['num'])
            markers[r][c] = num
        return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)

    @classmethod
    def fromFields(cls, rows, cols, heroRow, heroCol, heroDir, blocked, markers):
        return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)

    @classmethod
    def fromMatrix(cls, matrix):
        worldSize = int(math.sqrt(len(matrix) / MATRIX_DIMENSIONS))
        # this tensor is four dimensions:
        # dim 0: shows where the padding is
        # dim 1: shows where the obstacles are
        # dim 2: shows where the markers are
        # dim 3: shows where the hero is
        tensor = np.reshape(matrix, (MATRIX_DIMENSIONS, worldSize, worldSize))

        # get the dimension of the world
        padding = tensor[0]
        rows = 0
        cols = 0
        for r in range(worldSize):
            for c in range(worldSize):
                if padding[r][c] == 1:
                    rows = max(rows, r + 1)
                    cols = max(cols, c + 1)

        # get the blocked matrix
        blocked = np.zeros((rows,cols))
        for r in range(rows):
            for c in range(cols):
                if tensor[1][rows - r - 1][c] == 1:
                    blocked[r][c] = 1

        # get the markers matrix
        markers = np.zeros((rows,cols))
        for r in range(rows):
            for c in range(cols):
                markers[r][c] = tensor[2][rows - r - 1][c]

        heroPos = tensor[3]
        for r in range(worldSize):
            for c in range(worldSize):
                if heroPos[r][c] != 0:
                    heroRow = rows - r - 1
                    heroCol = c
                    heroDir = World.undoHeroDirValue(heroPos[r][c])
                    return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)
        raise Exception('no hero found')

    # Function: Equals
    # ----------------
    # Checks if two worlds are equal. Does a deep check.
    def __eq__(self, other):
        if self.heroRow != other.heroRow: return False
        if self.heroCol != other.heroCol: return False
        if self.heroDir != other.heroDir: return False
        if self.crashed != other.crashed: return False
        return self.equalMakers(other)

    def __ne__(self, other):
        return not (self == other)

    def hammingDist(self, other):
        dist = 0
        if self.heroRow != other.heroRow: dist += 1
        if self.heroCol != other.heroCol: dist += 1
        if self.heroDir != other.heroDir: dist += 1
        if self.crashed != other.crashed: dist += 1
        dist += np.sum(self.markers != other.markers)
        return dist

    # Function: Equal Markers
    # ----------------
    # Are the markers the same in these two worlds?
    def equalMakers(self, other):
        return (self.markers == other.markers).all()

    @staticmethod
    def parseJson(obj):
        rows = int(obj['rows'])
        cols = int(obj['cols'])

        hero_pos = obj['hero'].split(":")

        heroRow = int(hero_pos[0])
        heroCol = int(hero_pos[1])
        heroDir = str(hero_pos[2])

        blocked = np.zeros((rows, cols))
        if obj['blocked'] != "":
            for x in obj['blocked'].split(" "):
                pos = x.split(":")
                blocked[int(pos[0])][int(pos[1])] = 1

        markers = np.zeros((rows, cols))
        if obj['markers'] != "":
            for x in obj['markers'].split(" "):
                pos = x.split(":")
                markers[int(pos[0])][int(pos[1])] = int(pos[2])

        return World(rows, cols, heroRow, heroCol, heroDir, blocked, markers)

    def toJson(self):
        obj = {}

        obj['rows'] = self.rows
        obj['cols'] = self.cols
        if self.crashed:
            obj['crashed'] = True
            return obj

        obj['crashed'] = False

        markers = []
        blocked = []
        hero = []
        for r in xrange(self.rows-1, -1, -1):
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1:
                    blocked.append("{0}:{1}".format(r, c))
                if self.heroAtPos(r, c):
                    hero.append("{0}:{1}:{2}".format(r, c, self.heroDir))
                if self.markers[r][c] > 0:
                    markers.append("{0}:{1}:{2}".format(r, c, int(self.markers[r][c])))

        obj['markers'] = " ".join(markers)
        obj['blocked'] = " ".join(blocked)
        obj['hero'] = " ".join(hero)

        return obj

    # Function: toString
    # ------------------
    # Returns a string version of the world. Uses a '>'
    # symbol for the hero, a '*' symbol for blocked and
    # in the case of markers, puts the number of markers.
    # If the hero is standing ontop of markers, the num
    # markers is not visible.
    def toString(self):
        worldStr = ''
        #worldStr += str(self.heroRow) + ', ' + str(self.heroCol) + '\n'
        if self.crashed: worldStr += 'CRASHED\n'
        for r in range(self.rows-1, -1, -1):
            rowStr = '|'
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1 :
                    rowStr += '*'
                elif self.heroAtPos(r, c):
                    rowStr += self.getHeroChar()
                elif self.markers[r][c] > 0:
                    numMarkers = int(self.markers[r][c])
                    if numMarkers > 9: rowStr += 'M'
                    else: rowStr += str(numMarkers)
                else:
                    rowStr += ' '
            worldStr += rowStr + '|'
            if(r != 0): worldStr += '\n'
        return worldStr

    # Function: toTensor
    # ------------------
    # Returns a tensor version of the world which is 4xNxM where
    # N = numRows, M = numCols. The first grid is the padding, second is blocked state
    # the third grid is the beeper locations. The fourth grid is karel.
    # The rows of the grid are interchanged so the visualization of
    # karel will match the matrices. If the padding
    # parameter is not False, return a tensor which is 4xpaddingxpadding.
    def toTensor(self, padding):
        tensor = np.zeros([4, self.rows, self.cols])
        if padding:
            tensor = np.zeros([4, padding, padding])
        for r in range(self.rows):
            for c in range(self.cols):
                # make it so that the tensor looks like the visualization
                vizRow = self.rows - r - 1
                # first matrix shows the padding
                tensor[0][vizRow][c] = 1
                # second matrix is obstacles
                tensor[1][vizRow][c] = self.blocked[r][c]
                # third matrix is num markers
                tensor[2][vizRow][c] = self.markers[r][c]
        # fourth matri is the location of karel
        tensor[3][self.rows - self.heroRow - 1][self.heroCol] = self.getHeroDirValue()
        return tensor

    def toPytorchTensor(self, padding):
        tensor = torch.FloatTensor(16, padding, padding).zero_()
        # Put in the hero
        tensor[self.getHeroDirValue()-1][self.heroRow+1][self.heroCol+1] = 1

        # Put in the markers that serves to limit the padding
        # Vertical line
        for r in range(self.rows+2):
            tensor[5][r][0] = 1
            tensor[5][r][self.cols+1] = 1
        # Horizontal line
        for c in range(self.cols+2):
            tensor[5][0][c] = 1
            tensor[5][self.rows+1][c] = 1

        # Put in the obstacles
        for r in range(self.rows):
            for c in range(self.cols):
                # Obstacles
                tensor[4][r+1][c+1] = self.blocked[r][c]
                # Markers
                nb_markers = self.markers[r][c]
                if nb_markers > 0:
                    tensor[5+nb_markers][r+1][c+1] = 1

        return tensor

    @classmethod
    def fromPytorchTensor(cls, tensor):
        # Identify rows and cols
        nb_row = 0
        while nb_row < tensor.size(1) and tensor[5][nb_row][0] == 1:
            nb_row += 1
        rows = nb_row - 2

        nb_col = 0
        while nb_col < tensor.size(2) and tensor[5][0][nb_col] == 1:
            nb_col += 1
        cols = nb_col - 2

        # Get the obstacle matrix
        blocked = tensor[4, 1:rows+1, 1:cols+1].numpy()

        # Get the position of the hero
        hero_pos_val = torch.nonzero(tensor[:4]).squeeze()
        heroDir = cls.undoHeroDirValue(hero_pos_val[0]+1)
        heroRow = hero_pos_val[1] - 1
        heroCol = hero_pos_val[2] - 1

        # Get the position of the markers
        markers = np.zeros((rows, cols))
        for nb_marker_m1, marker_map in enumerate(tensor[6:,1:rows+1, 1:cols+1]):
            markers += (nb_marker_m1+1) * marker_map

        return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)



    # Function: get hero char
    # ------------------
    # Returns a char that represents the hero (based on
    # the heros direction).
    def getHeroChar(self):
        if(self.heroDir == 'north'): return '^'
        if(self.heroDir == 'south'): return 'v'
        if(self.heroDir == 'east'): return '>'
        if(self.heroDir == 'west'): return '<'
        raise("invalid dir")

    # Function: get hero dir value
    # ------------------
    # Returns a numeric representation of the hero direction.
    def getHeroDirValue(self):
        if(self.heroDir == 'north'): return 1
        if(self.heroDir == 'south'): return 3
        if(self.heroDir == 'east'): return 2
        if(self.heroDir == 'west'): return 4
        raise("invalid dir")

    @classmethod
    def undoHeroDirValue(cls, value):
        if(value == 1): return 'north'
        if(value == 3): return 'south'
        if(value == 2): return 'east'
        if(value == 4): return 'west'
        raise('invalid dir')

    # Function: hero at pos
    # ------------------
    # Returns true or false if the hero is at a given location.
    def heroAtPos(self, r, c):
        if self.heroRow != r: return False
        if self.heroCol != c: return False
        return True

    def isCrashed(self):
        return self.crashed

    # Function: is clear
    # ------------------
    # Returns if the (r,c) is a valid and unblocked pos.
    def isClear(self, r, c):
        if(r < 0 or c < 0):
            return False
        if r >= self.rows or c >= self.cols:
            return False
        if self.blocked[r][c] != 0:
            return False
        return True

    # Function: front is clear
    # ------------------
    # Returns if the hero is facing an open cell.
    def frontIsClear(self):
        if self.crashed: return
        self.noteApiCall()
        if(self.heroDir == 'north'):
            return self.isClear(self.heroRow + 1, self.heroCol)
        elif(self.heroDir == 'south'):
            return self.isClear(self.heroRow - 1, self.heroCol)
        elif(self.heroDir == 'east'):
            return self.isClear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'west'):
            return self.isClear(self.heroRow, self.heroCol - 1)


    # Function: left is clear
    # ------------------
    # Returns if the left of the hero is an open cell.
    def leftIsClear(self):
        if self.crashed: return
        self.noteApiCall()
        if(self.heroDir == 'north'):
            return self.isClear(self.heroRow, self.heroCol - 1)
        elif(self.heroDir == 'south'):
            return self.isClear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'east'):
            return self.isClear(self.heroRow + 1, self.heroCol)
        elif(self.heroDir == 'west'):
            return self.isClear(self.heroRow - 1, self.heroCol)


    # Function: right is clear
    # ------------------
    # Returns if the right of the hero is an open cell.
    def rightIsClear(self):
        if self.crashed: return
        self.noteApiCall()
        if(self.heroDir == 'north'):
            return self.isClear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'south'):
            return self.isClear(self.heroRow, self.heroCol - 1)
        elif(self.heroDir == 'east'):
            return self.isClear(self.heroRow - 1, self.heroCol)
        elif(self.heroDir == 'west'):
            return self.isClear(self.heroRow + 1, self.heroCol)


    # Function: markers present
    # ------------------
    # Returns if there is one or more markers present at
    # the hero pos
    def markersPresent(self):
        return self.markers[self.heroRow][self.heroCol] > 0
        self.noteApiCall()

    # Function: pick marker
    # ------------------
    # If there is a marker, pick it up. Otherwise crash the
    # program.
    def pickMarker(self):
        if not self.markersPresent():
            self.crashed = True
        else:
            self.markers[self.heroRow][self.heroCol] -= 1
        self.noteApiCall()

    # Function: put marker
    # ------------------
    # Adds a marker to the current location (can be > 1)
    def putMarker(self):
        self.markers[self.heroRow][self.heroCol] += 1
        if self.markers[self.heroRow][self.heroCol] > MAX_MARKERS_PER_SQUARE:
            self.crashed = True
        self.noteApiCall()

    # Function: move
    # ------------------
    # Move the hero in the direction she is facing. If the
    # world is not clear, the hero's move is undone.
    def move(self):
        if self.crashed: return
        newRow = self.heroRow
        newCol = self.heroCol
        if(self.heroDir == 'north'): newRow = self.heroRow + 1
        if(self.heroDir == 'south'): newRow = self.heroRow - 1
        if(self.heroDir == 'east'): newCol = self.heroCol + 1
        if(self.heroDir == 'west'): newCol = self.heroCol - 1
        if not self.isClear(newRow, newCol):
            self.crashed = True
        if not self.crashed:
            self.heroCol = newCol
            self.heroRow = newRow
        self.noteApiCall()

    def executeAction(self, actionString):
        action_func = getattr(self, actionString)
        action_func()

    # Function: turn left
    # ------------------
    # Rotates the hero counter clock wise.
    def turnLeft(self):
        if self.crashed: return
        if(self.heroDir == 'north'): self.heroDir = 'west'
        elif(self.heroDir == 'south'): self.heroDir = 'east'
        elif(self.heroDir == 'east'): self.heroDir = 'north'
        elif(self.heroDir == 'west'): self.heroDir = 'south'
        self.noteApiCall()

    # Function: turn left
    # ------------------
    # Rotates the hero clock wise.
    def turnRight(self):
        if self.crashed: return
        if(self.heroDir == 'north'): self.heroDir = 'east'
        elif(self.heroDir == 'south'): self.heroDir = 'west'
        elif(self.heroDir == 'east'): self.heroDir = 'south'
        elif(self.heroDir == 'west'): self.heroDir = 'north'
        self.noteApiCall()

    # Function: note api call
    # ------------------
    # To catch infinite loops, we limit the number of API calls.
    # If the num api calls exceeds a max, the program is crashed.
    def noteApiCall(self):
        self.numAPICalls += 1
        if self.numAPICalls > MAX_API_CALLS:
            self.crashed = True
