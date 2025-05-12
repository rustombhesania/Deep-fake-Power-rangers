# Summary this week’s Work 12th May 2025

## 1. **Padding and Sample Size vs. Sampling Rate**
   - **Padding**: In audio processing, padding ensures that all audio samples in a dataset have the same length. This is crucial for batch processing in neural networks where the input dimensions need to be consistent. 
     - If an audio file is shorter than the defined `sample_size`, we pad it with zeros (using `torch.nn.functional.pad`).
     - If it’s longer than the `sample_size`, we truncate it to ensure uniform length across all samples.

   - **Sample Size vs. Sampling Rate**:
     - The `sample_size` is the number of audio samples you want to process, often set to a value corresponding to the duration of the audio (e.g., 16000 samples for 1 second of audio at 16 kHz).
     - The **sampling rate** determines how often the audio waveform is sampled per second (e.g., 16 kHz means 16,000 samples per second). 
     - **Key Point**: The `sample_size` should align with the length of time you want to analyze from the audio. For instance, if you want to analyze 1 second of audio at 16 kHz, the `sample_size` should be 16000.

---

## 2. **MFCC Calculation with Librosa vs. Torchaudio**

### MFCC with **Librosa**:
   - **Librosa** is a popular audio analysis library in Python, especially for research and prototyping.
   - To compute MFCCs using **Librosa**, the typical steps are:
     1. Load the audio with `librosa.load()`.
     2. Pre-emphasis (optional) to filter high-frequency noise.
     3. Use `librosa.feature.mfcc()` to compute MFCC features.
     4. Apply **Mel-scale** and **Fourier Transform**.
     5. The result is usually in the form of a 2D matrix (MFCC coefficients for each frame).
   
   - **Advantages**:
     - **Flexible**: Many options for feature extraction and preprocessing (e.g., noise reduction, pre-emphasis).
     - **Customizable**: Easy to tweak parameters like window size and hop length.

   - **Disadvantages**:
     - **Slower Execution**: As it is more of a research-focused library, it's not as optimized for large-scale, GPU-based training.
     - **Manual Device Management**: To leverage GPUs, you'd have to manually move data to and from the device, which adds overhead.

### MFCC with **Torchaudio**:
   - **Torchaudio** is a library built on top of PyTorch that integrates seamlessly with the PyTorch ecosystem, providing faster operations and GPU support.
   - To compute MFCCs with **Torchaudio**, we use the `torchaudio.transforms.MFCC` transform:
     1. Load the audio using `torchaudio.load()`.
     2. Apply the `MFCC` transform directly to the waveform.
     3. The transform internally handles **Mel-filter banks**, **Fourier Transform**, and other necessary steps.
   
   - **Advantages**:
     - **GPU Accelerated**: It integrates well with PyTorch and can leverage GPU acceleration for faster computation.
     - **End-to-End Workflow**: Directly integrates into deep learning workflows, making it easier to implement in a neural network.
     - **Pre-Configured**: Many useful transformations are pre-configured, reducing manual work.

   - **Disadvantages**:
     - **Fewer Customization Options**: While efficient, it doesn't offer the same level of flexibility as Librosa for audio processing tasks.
     - **Limited Features**: Torchaudio is more focused on neural network training and may not have all the audio analysis tools found in Librosa.

---

## 3. **Handling CUDA Errors and Device-Side Assertions**
   - **CUDA Errors**: When working with GPU-based operations in PyTorch, you might encounter errors such as the `device-side assert triggered`. This error occurs when there’s an issue with the data or operations on the GPU, such as invalid tensor shapes, out-of-bounds indexing, or incompatible data types.
   
   - **Debugging CUDA Errors**: 
     - Set the environment variable `CUDA_LAUNCH_BLOCKING=1` to force PyTorch to execute operations synchronously. This makes it easier to trace the exact location of the error by blocking the asynchronous kernel launches.
     - **Example**: 
       ```bash
       export CUDA_LAUNCH_BLOCKING=1
       ```
     - **Fixing the Issue**: Ensure that all tensors are correctly shaped and indexed. For classification tasks, ensure that your labels are within the valid range (e.g., for a binary classification, the labels should be either `0` or `1`).

---

## 4. **Data Handling and Model Training Updates**
   - **Skipping Invalid Data**: If an error occurs while loading an audio file, we now skip the invalid files during training instead of using `-1` as the label.
   
   - **Training Loop**: During the training loop, we handle cases where `None` is returned (due to failed audio loading) by skipping those batches. This helps avoid training errors from invalid or missing data.

   - **Improved Accuracy Calculation**: 
     - We added accuracy calculation alongside the loss during training. This gives us a clearer indication of model performance in addition to the loss.
     - **Example**: The accuracy is computed as:
       ```python
       _, predicted = torch.max(outputs, 1)
       correct_preds += (predicted == y).sum().item()
       total_preds += y.size(0)
       accuracy = 100 * correct_preds / total_preds
       ```

---

## Conclusion

We explored and improved the audio processing pipeline by:
- Understanding the importance of **padding** and **sample size** for consistent input dimensions.
- Comparing **MFCC computation** using **Librosa** and **Torchaudio**, with a preference for **Torchaudio** in deep learning workflows due to its GPU support and PyTorch integration.
- Handling **CUDA errors** by debugging device-side assertions and ensuring tensor shapes are compatible.
- Improving **data handling** by skipping invalid files and ensuring that the model receives correct input during training.

This setup lays the foundation for training an **audio classification model** using **MFCC features** and **deep neural networks**.

