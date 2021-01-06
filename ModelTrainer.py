import glob
import os
from sklearn.preprocessing import normalize
import numpy as np
import re
import magenta.music as mm
import magenta
import tensorflow
from magenta.music import midi_io
from magenta.music import musicxml_reader
from magenta.music import musicxml_parser
from magenta.music import chords_lib
from magenta.music import note_sequence_io
import magenta.pipelines as mp
from magenta.music import sequences_lib
from magenta.pipelines import melody_pipelines
from magenta.music import melodies_lib
from magenta.pipelines import chord_pipelines

class ModelTrainer:

    def __init__(self):
        self.ct_matrix = np.zeros((25, 25))
        self.mo_matrix = np.zeros((25, 12))

    def cleanDataset(self):

        path = "D:\FAI\Wikifonia"
        count = 0
        for file in glob.glob(path):
            try:
                mxlObject = musicxml_parser.MusicXMLDocument(file)
                mxlSequence = musicxml_reader.musicxml_to_sequence_proto(mxlObject)
                quantizedNoteSequence = sequences_lib.quantize_note_sequence(mxlSequence, 1)
                chord_prog, stats = chord_pipelines.extract_chords(quantizedNoteSequence)
                melodies, stats = melody_pipelines.extract_melodies(quantizedNoteSequence)
                ac, stats = chord_pipelines.extract_chords_for_melodies(quantizedNoteSequence, melodies)
            except:
                os.remove(file)
                print(file)
                count = count + 1

    def observationModelTrainer(self):
        ALL_CHORD_LIST = ['N.C', 'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'Eb', 'Ebm', 'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G',
                          'Gm', 'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm']
        Same_Chord = {'Db': 'C#', 'Dbm': 'C#m', 'D#': 'Eb', 'D#m': 'Ebm', 'Gb': 'F#', 'Gbm': 'F#m', 'Ab': 'G#',
                      'Abm': 'G#m', 'Bb': 'A#', 'Bbm': 'A#m'}
        path = "D:\FAI\Wikifonia"
        for file in glob.glob(path):
            mxlObject = musicxml_parser.MusicXMLDocument(file)
            mxlSequence = musicxml_reader.musicxml_to_sequence_proto(mxlObject)
            quantizedNoteSequence = sequences_lib.quantize_note_sequence(mxlSequence, 1)

            chord_prog, stats = chord_pipelines.extract_chords(quantizedNoteSequence)
            previous = None
            for chord in list(chord_prog[0]):
                if previous is None:
                    previous = chord
                    continue

                curChord = re.sub(r'\d+', '', chord)
                prevChord = re.sub(r'\d+', '', previous)
                curChord = curChord[:3]
                prevChord = prevChord[:3]

                if curChord != 'N.C':
                    if len(curChord) == 3 and curChord[2] != 'm':
                        curChord = curChord[:2]
                        if curChord[1] not in ['#', 'b']:
                            curChord = curChord[:1]

                if prevChord != 'N.C':
                    if len(prevChord) == 3 and prevChord[2] != 'm':
                        prevChord = prevChord[:2]
                        if prevChord[1] not in ['#', 'b']:
                            prevChord = prevChord[:1]

                    if curChord in Same_Chord:
                        curChord = Same_Chord[curChord]
                    if prevChord in Same_Chord:
                        prevChord = Same_Chord[prevChord]

                    if curChord == 'Cb':
                        curChord = 'B'
                    if prevChord == 'Cb':
                        prevChord = 'B'
                    if curChord == 'Fb':
                        curChord = 'E'
                    if prevChord == 'Fb':
                        prevChord = 'E'
                    if curChord == 'Cbm':
                        curChord = 'D'
                    if prevChord == 'Cbm':
                        prevChord = 'D'
                    if curChord == 'Fbm':
                        curChord = 'Em'
                    if prevChord == 'Fbm':
                        prevChord = 'Em'

                    if prevChord != curChord:
                        a = ALL_CHORD_LIST.index(prevChord)
                        b = ALL_CHORD_LIST.index(curChord)
                        self.ct_matrix[a][b] = self.ct_matrix[a][b] + 1
                    previous = curChord

        normed_ct_matrix = normalize(self.ct_matrix, axis=1, norm='l1')
        self.ct_matrix = normed_ct_matrix

    def emissionModelTrainer(self):
        ALL_CHORD_LIST = ['N.C', 'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'Eb', 'Ebm', 'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G',
                          'Gm', 'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm']
        Same_Chord = {'Db': 'C#', 'Dbm': 'C#m', 'D#': 'Eb', 'D#m': 'Ebm', 'Gb': 'F#', 'Gbm': 'F#m', 'Ab': 'G#',
                      'Abm': 'G#m', 'Bb': 'A#', 'Bbm': 'A#m'}
        ALL_NOTE_LIST = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


        path = "D:\FAI\Wikifonia"
        for file in glob.glob(path):
            mxlObject = musicxml_parser.MusicXMLDocument(file)
            mxlSequence = musicxml_reader.musicxml_to_sequence_proto(mxlObject)
            quantizedNoteSequence = sequences_lib.quantize_note_sequence(mxlSequence, 1)

            melodies, stats = melody_pipelines.extract_melodies(quantizedNoteSequence)

            chord_prog, stats = chord_pipelines.extract_chords_for_melodies(quantizedNoteSequence, melodies)
            if not chord_prog:
                continue
            for i in range(len(list(chord_prog[0]))):
                curChord = list(chord_prog[0])[i]
                curMel = list(melodies[0])[i]
                while (curMel > 71):
                    curMel = curMel - 12
                while (curMel < 60):
                    curMel = curMel + 12
                curChord = re.sub(r'\d+', '', curChord)
                curChord = curChord[:3]
                if curChord not in 'N.C.':
                    if len(curChord) == 3 and curChord[2] not in 'm':
                        curChord = curChord[:2]
                        if curChord[1] not in ['#', 'b']:
                            curChord = curChord[:1]

                if curChord in Same_Chord:
                    curChord = Same_Chord[curChord]

                if curChord in 'Cb':
                    curChord = 'B'

                if curChord in 'Fb':
                    curChord = 'E'

                if curChord in 'Cbm':
                    curChord = 'D'

                if curChord in 'Fbm':
                    curChord = 'Em'

                a = ALL_CHORD_LIST.index(re.sub(r'\d+', '', curChord))
                b = curMel
                self.mo_matrix[a][b - 60] = self.mo_matrix[a][b - 60] + 1

        normed_mo_matrix = normalize(self.mo_matrix, axis=1, norm='l1')
        self.mo_matrix = normed_mo_matrix

    def saveTrainedWeights(self):
        np.save("D:\FAI\ctmatrix", self.ct_matrix)
        np.save("D:\FAI\momatrix", self.mo_matrix)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.observationModelTrainer()
    trainer.emissionModelTrainer()
    trainer.saveTrainedWeights()

