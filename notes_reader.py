'''
Utility to read notes from a notes text file
'''

class NotesReader(object):
    def __init__(self):
        pass

    def read_notes_from_file(self, file_path):
        pass

    def combine_notes(self, notes):
        if len(notes) == 0:
            return
        
        combined = []
        
        for i in range(len(notes)):
            notes[i] = list(notes[i])
            notes[i].reverse()
            
        
        while len(notes[i]) > 0:
            items = []
            for note in notes:
                items.append(note.pop())
            appended = False
            for item in items:
                if item != '_':
                    combined.append(item)
                    appended = True
                    break
                    
            if not appended:
                combined.append('_')

        return combined


