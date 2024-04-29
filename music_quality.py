import muspy as mp
import os

# Scale Consistency sc
# Pitch Entropy pe
# Pitch Class Entropy pce
# Empty Beat Rate ebr
# drum pattern consistency dpc

'''This python file is to cal the music quality >>> music_q.txt'''
if __name__ == '__main__':

    target_dir_list = ["vmcpdata/30_midi", "vmcpdata/50_midi", "vmcpdata/simple_midi","vmcpdata/drama_midi"]
    

    for target_dir in target_dir_list:
        midi_len  = len(os.listdir(target_dir))
        sc =0.0
        pe =0.0
        pce = 0.0
        ebr = 0.0
        dpc = 0.0
        ct = 0
        for idx, filename in enumerate(os.listdir(target_dir)):
            
            ct+=1
            print(f'-----{ct} / {midi_len} -----')
            # print(ct)
            #read in midi
            file_path = os.path.join(target_dir, filename)
            music = mp.read_midi(file_path)

            id_sc = mp.scale_consistency(music)
            id_pe = mp.pitch_entropy(music)
            id_pce = mp.pitch_class_entropy(music)
            id_ebr = mp.empty_beat_rate(music)
            id_dpc = mp.drum_pattern_consistency(music)

            sc += id_sc
            pe += id_pe
            pce += id_pce
            ebr += id_ebr
            dpc += id_dpc

            

            print(f'id_sc = {id_sc},     id_pe= {id_pe},    id_pce= {id_pce},    id_ebr={id_ebr},    id_dpc = {id_dpc}')
        sc /= midi_len
        pe /= midi_len
        pce /= midi_len
        ebr /= midi_len
        dpc /= midi_len
        print(f'========================{target_dir}')
        print(f'sc = {sc}')
        print(f'pe= {pe}')
        print(f'pce= {pce}')
        print(f'ebr={ebr}')
        print(f'dpc = {dpc}')