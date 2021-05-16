# map the name of a hemisphere to CIFTI brain structure
hemi2stru = {
    'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}

# map roi names to their labels
roi2label = {
    'IOG-face': 1,
    'pFus-face': 2,
    'mFus-face': 3
}

# map roi names to colors
roi2color = {
    'IOG-face': 'red',
    'pFus-face': 'limegreen',
    'mFus-face': 'cornflowerblue'
}

# map Cole network name to label
# /nfs/p1/atlases/ColeAnticevicNetPartition/network_labelfile.txt
net2label_cole = {
    'Primary Visual': 1,
    'Secondary Visual': 2,
    'Somatomotor': 3,
    'Cingulo-Opercular': 4,
    'Dorsal-attention': 5,
    'Language': 6,
    'Frontoparietal': 7,
    'Auditory': 8,
    'Default': 9,
    'Posterior Multimodal': 10,
    'Ventral Multimodal': 11,
    'Orbito-Affective': 12
}
