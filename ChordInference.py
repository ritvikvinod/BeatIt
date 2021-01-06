import tensorflow as tf
import magenta as mm
import numpy as np
from sklearn.preprocessing import normalize
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.pipelines import melody_pipelines
from music21 import *
import os

class ChordInference:

    def __init__(self):
        ctMatrix = np.load("WeightedSameChordModel.npy")
        moMatrix = np.load("y.npy")
        self.chordTrans = ctMatrix
        self.melObs = moMatrix

    def generateChords(self,rounded_pitches):
        ALL_CHORD_LIST = ['N.C', 'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'Eb', 'Ebm', 'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G',
                          'Gm', 'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm']
        obsModel = self.chordTrans
        emsModel = self.melObs
        observations = rounded_pitches
        T = len(ALL_CHORD_LIST)
        M = len(rounded_pitches)
        initialP = np.full(25, 1 / 25)
        viterbi = np.zeros((T, M))
        print(viterbi)
        viterbi[:, 0] = np.log(initialP * emsModel[:, observations[0] - 60])

        prev = np.zeros((T, M - 1))
        for melody in range(1, M):
            for chord in range(T):

                allProbs = viterbi[:, melody - 1] + np.log(obsModel[:, chord]) + np.log(
                    emsModel[chord, observations[melody] - 60])
                prev[chord, melody - 1] = np.argmax(allProbs)
                viterbi[chord, melody] = np.max(allProbs)

        path = np.zeros(M)
        lastChord = np.argmax(viterbi[:, M - 1])
        path[0] = lastChord
        backtrackIndex = 1

        for i in range(M - 2, 0, -1):
            path[backtrackIndex] = prev[int(lastChord), i]
            lastChord = prev[int(lastChord), i]
            backtrackIndex = backtrackIndex + 1

        path = np.flip(path, axis=0)
        path = np.round(path)
        print(path)
        result = []
        for chord in path:
            result.append(ALL_CHORD_LIST[int(chord)])

        return result

    def extractPitch(self,song):
        seq = midi_io.midi_file_to_note_sequence(song)
        qseq = sequences_lib.quantize_note_sequence(seq, 1)
        melodies, stats = melody_pipelines.extract_melodies(qseq)
        rounded_pitches = []
        for mel in list(melodies[0]):
            while (mel > 71):
                mel = mel - 12
            while (mel < 60):
                mel = mel + 12
            rounded_pitches.append(mel)

        return rounded_pitches

    def generateOutputFile(self, result):
        ChordProgressions = {'C': ["C4", "E4", "G4"], 'Cm': ["C4", "E-4", "G4"],
                             'C#': ["C#4", "E#4", "G#4"], 'C#m': ["C#4", "E4", "G#4"],
                             'D': ["D4", "F#4", "A4"], 'Dm': ["D4", "F4", "A4"],
                             'Eb': ["E-4", "G4", "B-4"], 'Ebm': ["E-4", "G-4", "B-4"],
                             'E': ["E4", "G#4", "B4"], 'Em': ["E4", "G4", "B4"],
                             'F': ["F4", "A4", "C5"], 'Fm': ["F4", "A-4", "C5"],
                             'F#': ["F#4", "A#4", "C#5"], 'F#m': ["F#4", "A4", "C#5"],
                             'G': ["G4", "B4", "D5"], 'Gm': ["G4", "B-4", "D5"],
                             'G#': ["A-4", "C5", "E-5"], 'G#m': ["A-4", "B-4", "E-5"],
                             'A': ["A4", "C#5", "E5"], 'Am': ["A4", "C5", "E5"],
                             'A#': ["B-4", "D5", "F5"], 'A#m': ["B-4", "D-5", "F5"],
                             'B': ["B4", "D#5", "F#5"], 'Bm': ["B4", "D5", "F#5"],
                             'N.C': []}

        s = stream.Stream()
        for index, chords in enumerate(result):
            if (chords != 'N.C'):
                c = chord.Chord(ChordProgressions[chords], duration=duration.Duration(1))
                s.append(c)
            else:
                s.append(note.Rest())

        fp = s.write('midi', fp='samajavaragamana.mid')


if __name__== "__main__":
    #path = os.path.abspath("D:\FAI\heathens.midi")
    model = ChordInference()
    observations = model.extractPitch("insert path here")
    result = model.generateChords(observations)
    model.generateOutputFile(result)