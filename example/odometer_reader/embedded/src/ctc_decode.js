const NINF = -Infinity;
const DEFAULT_EMISSION_THRESHOLD = 0.01;

function reconstruct(labels, blank = 0) {
    const newLabels = [];
    let previous = null;
    
    for (const label of labels) {
        if (label !== previous) {
            newLabels.push(label);
            previous = label;
        }
    }
    
    return newLabels.filter(label => label !== blank);
}

function beamSearchDecode(emissionLogProb, blank = 0, beamSize = 10) {
    const emissionThreshold = Math.log(DEFAULT_EMISSION_THRESHOLD);
    const length = emissionLogProb.length;
    const classCount = emissionLogProb[0].length;
    
    let beams = [{ prefix: [], score: 0 }];
    
    for (let t = 0; t < length; t++) {
        const newBeams = [];
        
        for (const beam of beams) {
            for (let c = 0; c < classCount; c++) {
                const logProb = emissionLogProb[t][c];
                if (logProb < emissionThreshold) continue;
                
                newBeams.push({
                    prefix: [...beam.prefix, c],
                    score: beam.score + logProb
                });
            }
        }
        
        // Sort and keep top beam_size beams
        newBeams.sort((a, b) => b.score - a.score);
        beams = newBeams.slice(0, beamSize);
    }
    
    return reconstruct(beams[0].prefix, blank);
}

module.exports = { beamSearchDecode };