# 1004 = lh-corpuscallosum
lh_aparc = [[aparc] for aparc in range(1001, 1036) if aparc != 1004]
left_areas = [[2],
              [4],
              [5],
              [6,7,8],
              [9,10],
              [11],
              [12],
              [13],
              [17],
              [18],
              [26],
              [28],
              [31]] + lh_aparc

left_labels = [x for y in left_areas for x in y]

# 2004 = rh-corpuscallosum
rh_aparc = [[aparc] for aparc in range(2001, 2036) if aparc != 2004]
right_areas = [[41],
               [43],
               [44],
               [45,46,47],
               [48,49],
               [50],
               [51],
               [52],
               [53],
               [54],
               [58],
               [60],
               [63]] + rh_aparc

right_labels = [x for y in right_areas for x in y]

target_label_sets = [[2],
                     [4,5,31],
                     [6,7,8],
                     [9,10],
                     [11],
                     [12],
                     [13],
                     [17],
                     [18],
                     [26],
                     [28],
                     [41],
                     [43,44,63],
                     [45,46,47],
                     [48,49],
                     [50],
                     [51],
                     [52],
                     [53],
                     [54],
                     [58],
                     [60],
                     [16],
                     [14],
                     [15],
                     [251,252,253,254,255]] + lh_aparc + rh_aparc + [
                     [x for l in lh_aparc for x in l],
                     [x for r in rh_aparc for x in r],
                    ]

all_labels = [x for y in target_label_sets for x in y]+[77]

non_lateral_labels = [x for x in all_labels if x not in left_labels and x not in right_labels]

target_label_sets.append(left_labels)
target_label_sets.append(right_labels)
target_label_sets.append(all_labels)

aparc_labels = ['bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus','entorhinal',
                'fusiform','inferiorparietal','inferiortemporal','isthmuscingulate','lateraloccipital',
                'lateralorbitofrontal','lingual','medialorbitofrontal','middletemporal','parahippocampal',
                'paracentral','parsopercularis','parsorbitalis','parstriangularis','pericalcarine','postcentral',
                'posteriorcingulate','precentral','precuneus','rostralanteriorcingulate','rostralmiddlefrontal',
                'superiorfrontal','superiorparietal','superiortemporal','supramarginal','frontalpole',
                'temporalpole','transversetemporal','insula']

target_label_names = ['Left-Cerebral-White-Matter', 'Left-Ventricle-all',
                      'Left-Cerebellum', 'Left-Thalamus-Proper', 'Left-Caudate',
                      'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 
                      'Left-Accumbens-area', 'Left-VentralDC', 
                      'Right-Cerebral-White-Matter', 'Right-Ventricle-all',
                      'Right-Cerebellum', 'Right-Thalamus-Proper', 'Right-Caudate',
                      'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 
                      'Right-Accumbens-area', 'Right-VentralDC', 'Brain-Stem','3rd-Ventricle','4th-Ventricle',
                      'Corpus-Callosum'
                      ]+['lh-{}'.format(l) for l in aparc_labels]+['rh-{}'.format(l) for l in aparc_labels]+[
                      'Left-Cerebral-Cortex',
                      'Right-Cerebral-Cortex',
                      'left-hemisphere',
                      'right-hemishpere',
                      'brain']

# Left/Right-Cerebral-Cortex, left/right-hemisphere, brain                    
NUM_IGNORE_LABELS=5
