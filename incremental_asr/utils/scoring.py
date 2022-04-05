import jiwer 

def calculate_batch_wer(refs, hypos):
    assert len(refs) == len(hypos)
    
    all_wer = 0
    for ref, hyp in zip(refs, hypos):
        wer = jiwer.wer(ref, hyp)
        all_wer += wer
    return all_wer / len(refs)