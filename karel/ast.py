import json

# Class: AST
# ----------
# An AST is stored as JSON :)

class Ast():
	def __init__(self, j_ast, guid=None):
		self.guid = guid
		if 'guid' in j_ast:
			self.guid = j_ast['guid']
			del j_ast['guid']
		self.astJson = j_ast

	def getJson(self):
		return self.astJson

	def getGuid(self):
		return self.guid

	def toString(self):
		return json.dumps(self.astJson, sort_keys=True, indent=2)

	def toPython(self, crashProtection = False):
		return PythonTranslator().toPython(self, crashProtection)

	def numType(self, typeName):
		total = 0
		for methodName in self.astJson:
			methodJson = self.astJson[methodName]
			total = self._numTypeBlock(methodJson, typeName)
		return total

	def _numTypeBlock(self, blockJson, typeName):
		count = 0
		for block in blockJson:
			blockType = block['type']
			if blockType == typeName:
				count += 1

			# For loops
			elif blockType == 'repeat':
				body = block['body']
				count += self._numTypeBlock(body, typeName)

			# While loops
			elif blockType == 'while':
				body = block['body']
				count += self._numTypeBlock(body, typeName)

			# If statements
			elif blockType == 'if':
				body = block['body']
				count += self._numTypeBlock(body, typeName)

			# If/else statements
			elif blockType == 'ifElse':
				ifBody = block['ifBody']
				elseBody = block['elseBody']
				count += self._numTypeBlock(ifBody, typeName)
				count += self._numTypeBlock(elseBody, typeName)

		return count



