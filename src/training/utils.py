from collections import defaultdict
from typing import List, Dict, Tuple, Callable, Any

import torch
import numpy as np
from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01



def labels_to_string(labels, label2char):
    return ''.join([label2char[label] for label in labels if label in label2char])


def _reconstruct(labels: List[int], blank: int = 0) -> List[int]:
    """Reconstruct a sequence by merging consecutive duplicate labels and removing blanks.

    This function processes a list of labels (e.g., the output of a CTC decoding step) by:
    1. **Merging duplicates**: Consecutive identical labels are collapsed into a single instance. 
       For example, `[1, 1, 2, 2, 2, 3]` becomes `[1, 2, 3]`.
    2. **Removing blanks**: The blank label (typically representing silence or no prediction in CTC) 
       is removed from the sequence. For example, `[1, 0, 2, 0, 3]` becomes `[1, 2, 3]`.

    This process is crucial for transforming raw CTC outputs into human-readable or usable sequences.

    **How it works**:
    - Iterate through the input list of labels.
    - Append a label to the output only if it differs from the previous one, ensuring duplicates are merged.
    - Filter out any instances of the blank label in a second pass.

    **Why it's needed**:
    - In CTC-based models, the output often contains repeated labels due to the model's probabilistic nature.
    - The blank label is used to indicate "no prediction" or a separator between repeated labels.
    - This function cleans and simplifies the sequence while preserving its meaning.

    Args:
        labels (List[int]): List of integer labels, which may contain duplicates and blanks.
        blank (int, optional): The index of the blank label in the class set. Defaults to 0.

    Returns:
        List[int]: The reconstructed sequence with duplicates merged and blanks removed.

    **Example**:
        >>> labels = [1, 1, 0, 2, 2, 0, 3, 3, 3]
        >>> reconstructed = _reconstruct(labels, blank=0)
        >>> print(reconstructed)
        [1, 2, 3]

    **Performance**:
        - Time Complexity: O(n), where `n` is the length of the input list. The function performs 
          a single traversal of the list, followed by a filtering step.
        - Space Complexity: O(n), as a new list is created for the output.

    Notes:
        - The function assumes the input labels are integers.
        - It works with any integer value for `blank` as long as it is consistent with the model's configuration.
    """
    new_labels = []
    previous = None

    # Merge consecutive duplicates
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # Remove blank labels
    new_labels = [l for l in new_labels if l != blank]

    return new_labels



def greedy_decode(emission_log_prob: np.ndarray, blank: int = 0, **kwargs: Any) -> List[int]:
    """Greedy decoding for CTC.

    Greedy decoding selects the label with the highest probability at each time step. 
    It is a simple and computationally efficient method, but it does not consider 
    alternative paths or the overall sequence probability.

    **How it works**:
    1. At each time step `t`, the class with the highest log probability is selected.
    2. Repeated labels are merged into one, and blanks are removed. For example:
       - Input sequence: `[blank, A, A, blank, B, B, blank]`
       - Output sequence: `[A, B]`
    3. The method assumes the highest-probability label at each step contributes 
       to the optimal sequence, which may not always be true in ambiguous cases.

    Args:
        emission_log_prob (np.ndarray): Emission log probabilities of shape `(T, C)`, 
            where `T` is the sequence length and `C` is the number of classes (including the blank class).
        blank (int, optional): Index of the blank label in the class set. Defaults to 0.

    Returns:
        List[int]: Decoded labels after collapsing repetitions and removing blanks.

    Example:
        >>> emission_log_prob = np.log([[0.1, 0.6, 0.3], 
                                        [0.1, 0.6, 0.3], 
                                        [0.7, 0.2, 0.1], 
                                        [0.1, 0.6, 0.3]])
        >>> decoded = greedy_decode(emission_log_prob, blank=0)
        >>> print(decoded)
        [1]
    """
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def beam_search_decode(emission_log_prob: np.ndarray, blank: int = 0, **kwargs: Any) -> List[int]:
    """Beam search decoding for CTC.

    Beam search decoding keeps track of the `beam_size` most probable sequences at each time step. 
    Instead of greedily selecting the most probable label at each step, this method explores multiple 
    paths and selects the one with the highest cumulative log probability across the sequence.

    **How it works**:
    1. **Initialization**:
       - Start with an empty sequence (prefix) and an accumulated log probability of 0.
       - Represent the beams as a list of tuples `(prefix, accumulated_log_prob)`.
    
    2. **Iterative Expansion**:
       - For each time step `t`:
           - Compute the log probability of each class at that time step.
           - Extend each sequence in the current beam with all possible classes (`C`).
           - For each extended sequence, update the cumulative log probability by adding the current log probability.

    3. **Beam Pruning**:
       - Sort the extended beams by their cumulative log probability in descending order.
       - Keep only the top `beam_size` sequences for the next iteration.

    4. **Finalization**:
       - At the end of all time steps, merge repeated labels and remove blanks.
       - Return the sequence with the highest cumulative log probability.

    **Advantages**:
    - By exploring multiple paths, beam search can recover from local errors in the emission probabilities.
    - It balances computational efficiency with accuracy, depending on the value of `beam_size`.

    **Limitations**:
    - Larger `beam_size` values increase computational cost.
    - May still miss the optimal sequence if `beam_size` is too small.

    Args:
        emission_log_prob (np.ndarray): Emission log probabilities of shape `(T, C)`, 
            where `T` is the sequence length and `C` is the number of classes (including the blank class).
        blank (int, optional): Index of the blank label in the class set. Defaults to 0.
        beam_size (int): Number of beams to maintain at each time step.
        emission_threshold (float, optional): Log probability threshold below which classes are ignored. 
            Defaults to `np.log(DEFAULT_EMISSION_THRESHOLD)`.

    Returns:
        List[int]: Decoded labels after applying beam search decoding.

    Example:
        >>> emission_log_prob = np.log([[0.1, 0.6, 0.3], 
                                        [0.1, 0.6, 0.3], 
                                        [0.7, 0.2, 0.1], 
                                        [0.1, 0.6, 0.3]])
        >>> decoded = beam_search_decode(emission_log_prob, blank=0, beam_size=2)
        >>> print(decoded)
        [1]
    """
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        total_accu_log_prob[labels] = logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob) for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob: np.ndarray, blank: int = 0, **kwargs: Any) -> List[int]:
    """Prefix beam search decoding for CTC.

    Prefix beam search is an optimized version of beam search that considers shared prefixes of sequences. 
    This method maintains separate log probabilities for sequences ending in a blank and those not ending in a blank, 
    allowing for more efficient pruning and better management of sequences with high redundancy.

    **How it works**:
    1. **Initialization**:
       - Start with an empty sequence (prefix) and initialize its probabilities:
         - `log_prob_blank`: Probability of the sequence ending in a blank.
         - `log_prob_non_blank`: Probability of the sequence not ending in a blank.

    2. **Iterative Expansion**:
       - For each time step `t`, update the probabilities for all prefixes:
           - **Blank extension**: Extend the sequence with a blank, which preserves the current prefix.
           - **Non-blank extension**: Extend the sequence with a non-blank class. This creates new prefixes.
           - Use `logsumexp` to combine probabilities where applicable (e.g., when multiple paths lead to the same prefix).

    3. **Beam Pruning**:
       - Sort prefixes by their total log probability (sum of blank and non-blank probabilities).
       - Keep only the top `beam_size` prefixes for the next iteration.

    4. **Finalization**:
       - At the end of all time steps, the most probable sequence is the one with the highest total log probability.
       - Merge repeated labels and remove blanks.

    **Advantages**:
    - Efficiently manages prefixes, avoiding redundant computations for paths with the same prefix.
    - Provides better accuracy compared to standard beam search in scenarios with repeated labels or blanks.

    **Limitations**:
    - Higher computational complexity than greedy decoding.
    - Performance depends on the choice of `beam_size`.

    Args:
        emission_log_prob (np.ndarray): Emission log probabilities of shape `(T, C)`, 
            where `T` is the sequence length and `C` is the number of classes (including the blank class).
        blank (int, optional): Index of the blank label in the class set. Defaults to 0.
        beam_size (int): Number of beams to maintain at each time step.
        emission_threshold (float, optional): Log probability threshold below which classes are ignored. 
            Defaults to `np.log(DEFAULT_EMISSION_THRESHOLD)`.

    Returns:
        List[int]: Decoded labels after applying prefix beam search decoding.

    Example:
        >>> emission_log_prob = np.log([[0.1, 0.6, 0.3], 
                                        [0.1, 0.6, 0.3], 
                                        [0.7, 0.2, 0.1], 
                                        [0.1, 0.6, 0.3]])
        >>> decoded = prefix_beam_decode(emission_log_prob, blank=0, beam_size=2)
        >>> print(decoded)
        [1]
    """
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob))

    for t in range(length):
        new_beams_dict = defaultdict(lambda: (NINF, NINF))

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                new_prefix = prefix + (c,)
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    return labels


def ctc_decode(log_probs: torch.Tensor, label2char: Dict[int, str] = None, blank: int = 0,
               method: str = 'beam_search', beam_size: int = 10) -> List[List[int]]:
    """CTC decoding for sequence-to-sequence models.

    This function decodes the log probabilities output by a model trained with the 
    Connectionist Temporal Classification (CTC) loss function. The decoding process 
    transforms these log probabilities into the most likely sequences of labels 
    using one of the supported decoding methods.

    **How it works**:
    - CTC models output a matrix of shape `(T, C)` for each sequence, where `T` is 
      the number of time steps and `C` is the number of classes (including the blank class). 
    - The model produces probabilities for each class at each time step, but the 
      predictions may contain repeated labels or blanks.
    - The goal of decoding is to infer the most likely sequence of labels while 
      collapsing repeated labels and removing blanks.

    Supported decoding methods:
    1. **Greedy decoding**: Selects the label with the highest probability at each time step.
       - Fastest method, but does not explore alternative paths.
       - May fail to find the optimal sequence if errors exist in the emission probabilities.

    2. **Beam search decoding**: Keeps track of the top `beam_size` most probable sequences.
       - Explores multiple paths to find the optimal sequence.
       - Balances exploration and efficiency.

    3. **Prefix beam search decoding**: Optimized version of beam search.
       - Maintains separate probabilities for sequences ending in a blank and those not ending in a blank.
       - Efficiently prunes unlikely paths by considering shared prefixes.

    Args:
        log_probs (torch.Tensor): Tensor of shape `(N, T, C)` representing log probabilities 
            output by the model, where `N` is the batch size, `T` is the sequence length, 
            and `C` is the number of classes (including the blank class).
        label2char (Dict[int, str], optional): Mapping from labels (integers) to characters (strings). 
            Used to convert decoded labels to readable text. Defaults to None.
        blank (int, optional): Index of the blank label in the class set. Defaults to 0.
        method (str, optional): Decoding method to use. Defaults to 'beam_search'.
            - 'greedy': Greedy decoding.
            - 'beam_search': Beam search decoding.
            - 'prefix_beam_search': Prefix beam search decoding.
        beam_size (int, optional): Beam size for the beam search decoding methods. Defaults to 10.

    Returns:
        List[List[int]]: List of decoded sequences for each sample in the batch. Each sequence 
        is represented as a list of integers (labels). If `label2char` is provided, sequences 
        are converted to characters.

    Example:
        >>> log_probs = torch.randn(2, 5, 10).log_softmax(dim=-1)
        >>> label2char = {0: '-', 1: 'A', 2: 'B', 3: 'C'}
        >>> decoded = ctc_decode(log_probs, label2char, method='greedy')
        >>> print(decoded)
        [['A', 'B'], ['C']]
    """

    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list