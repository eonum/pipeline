package ch.eonum.pipeline.classification.lstm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Entry;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.core.SparseSequence;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Sparse implementation of LSTMCore.
 * Input units are sparse. Everything else is not.
 * 
 * @author tim
 *
 * @param <E>
 */
public class SparseLSTMCore<E extends SparseSequence> extends LSTMCore<E> {
	
	/**
	 * matrix for storing the input per training element per sequence element
	 * per input component
	 */
	private Entry[][][] input;
	/**
	 * matrix for storing the input per test element per sequence element per
	 * input component
	 */
	private Entry[][][] inputTest;
	/**
	 * Current point in sequence.
	 */
	private Entry[] currentInput;

	public SparseLSTMCore(String threadName, String baseDir, LSTM<E> parent,
			int seed, boolean outputGates, boolean forgetGates,
			boolean inputGates, boolean dropout) {
		super(threadName, baseDir, parent, seed, outputGates, forgetGates, inputGates,
				dropout);
	}

	@Override
	protected void forwardPass(boolean testTime) {
		int i, j, u, v, k;
		double sum;

		/** hidden units */
		for (i = numInputs; i < numHiddenAndInput; i++) {
			sum = 0;
			for (j = 0; j < currentInput.length; j++)
				sum += W_mod[i][currentInput[j].index] * currentInput[j].value;
			for (j = numInputs; j < numInpHidCells; j++)
				sum += W_mod[i][j] * Yk_mod_old[j];
			Yk_mod_new[i] = 1 / (1 + Math.exp(-sum));
		}
		if (biasOut)
			Yk_mod_new[numHiddenAndInput - 1] = 1.0;

		/** ### memory cells ### */

		i = numHiddenAndInput - 1;
		for (u = 0; u < numBlocks; u++) {

			/** input gate */
			if (inGates) {
				i++;
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (j = 0; j < currentInput.length; j++)
						sum += W_mod[i][currentInput[j].index] * currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++)
						sum += W_mod[i][j] * Yk_mod_old[j];
					Y_in[u] = 1 / (1 + Math.exp(-sum));
					Yk_mod_new[i] = Y_in[u];
				}
			}

			/** output gate */
			if (outGates) {
				i++;
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (j = 0; j < currentInput.length; j++)
						sum += W_mod[i][currentInput[j].index] * currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++)
						sum += W_mod[i][j] * Yk_mod_old[j];
					Y_out[u] = 1 / (1 + Math.exp(-sum));
					Yk_mod_new[i] = Y_out[u];
				}
			}

			/** forget gate */
			if (forgetGates) {
				i++;
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (j = 0; j < currentInput.length; j++)
						sum += W_mod[i][currentInput[j].index] * currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++)
						sum += W_mod[i][j] * Yk_mod_old[j];
					Y_forget[u] = 1 / (1 + Math.exp(-sum));
					Yk_mod_new[i] = Y_forget[u];
				}
			}

			/** uth memory cell block */
			for (v = 0; v < blockSize[u]; v++) {
				/** activation of function g of vth memory cell of block u */
				i++;
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (j = 0; j < currentInput.length; j++)
						sum += W_mod[i][currentInput[j].index] * currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++)
						sum += W_mod[i][j] * Yk_mod_old[j];
					G[u][v] = 4.0 / (1 + Math.exp(-sum)) - 2.0;
					/** update internal state */
					double forget = forgetGates ? Y_forget[u] : 1.0;
					double input = inGates ? Y_in[u] : 1.0;
	
					S[u][v] = forget * S[u][v] + input * G[u][v];
					/** activation function h */
					H[u][v] = 2.0 / (1 + Math.exp(-S[u][v])) - 1.0;
					/** activation of vth memory cell of block u */
					if (outGates)
						Yc[u][v] = H[u][v] * Y_out[u];
					else
						Yc[u][v] = H[u][v];
					Yk_mod_new[i] = Yc[u][v];
				}
			}
		}

		/** ### output units activation ### */

		if (targetExists) /** only if target for this input */
		{
			for (k = numInpHidCells; k < numAll; k++) {
				/** hidden units input */
				sum = 0;
				for (i = numInputs; i < numHiddenAndInput; i++) {
					sum += W_mod[k][i] * Yk_mod_new[i];
				}
				/** memory cells input */
				i = numHiddenAndInput - 1;
				for (u = 0; u < numBlocks; u++) {
					if (inGates)
						i++;
					if (outGates)
						i++;
					if (forgetGates)
						i++;
					for (v = 0; v < blockSize[u]; v++) {
						i++;
						if(!dropout || dropouts[u])
							sum += W_mod[k][i] * Yk_mod_new[i];
					}
				}
				/** activation */
				if(softmax)
					Yk_mod_new[k] = Math.exp(sum);
				else
					Yk_mod_new[k] = 1 / (1 + Math.exp(-sum));
			}
			/**
			 * normalize all output units to sum up to 1 when using the softmax
			 * activation function.
			 */
			if(softmax){
				double total = 0.0;
				for (k = numInpHidCells; k < numAll; k++)
					total += Yk_mod_new[k];
				for (k = numInpHidCells; k < numAll; k++)
					Yk_mod_new[k] /= total;
			}
		}
	}

	/**
	 * Test the net on the validation set.
	 */
	@Override
	protected void testLSTM() {
		int i, k, j;
		numForecasts = 0;
		epochErr = 0;
		dropouts = new boolean[this.numBlocks];
		for(i = 0; i < dropouts.length; i++)
			dropouts[i] = true;
		if(dropout)
			multiplyMemoryCellWeights(0.5);

		for (int currentSequence = 0; currentSequence < inputTest.length; currentSequence++) {
			resetNet();
			for (int currentElement = 0; currentElement < inputTest[currentSequence].length; currentElement++) {
				setInput(inputTest[currentSequence][currentElement],
						targetTest[currentSequence][currentElement]);

				forwardPass(true);

				if (targetExists) /** only if target for this input */
				{
					/** compute error */
					for (k = numInpHidCells, j = 0; k < numAll; k++, j++)
						error[j] = target_a[j] - Yk_mod_new[k];

					/** Training error */
					epochErr += compError(currentSequence, true);
				}

				/** set old activations */
				for (i = numInputs; i < numAll; i++) {
					Yk_mod_old[i] = Yk_mod_new[i];
				}
			}

		}
		
		if(dropout)
			multiplyMemoryCellWeights(2.0);
		
		Log.printf("[LSTM-Thread: %s] TEST: epochs:%d sequences:%d\n",
				threadName, epoch + 1, numbSeq);
		Log.printf("[LSTM-Thread: %s] TEST: " + (this.softmax ? "cross entropy" : "RMSE") + ":%.6f\n", threadName,
				Math.sqrt(epochErr / numForecasts));

		Log.printf("\n");
	}

	/**
	 * feed the net with data.
	 * @param input
	 * @param target
	 */
	private void setInput(Entry[] input, double[] target) {
		targetExists = true;
		this.currentInput = input;
		for (int k = numInpHidCells, j = 0; k < numAll; k++, j++) {
			target_a[j] = target[j];
			if (Double.isNaN(target[j]))
				targetExists = false;
		}			
	}

	/**
	 *  Derivatives of the internal state over time.
	 */
	@Override
	protected void derivatives() {
		int u, v, j;
		for (u = 0; u < numBlocks; u++) {
			if(!dropout || dropouts[u])
				for (v = 0; v < blockSize[u]; v++) {
					/** weights to input gate */
					if (inGates) {
						for (j = 0; j < currentInput.length; j++)
							SI[u][v][currentInput[j].index] = SI[u][v][currentInput[j].index]
									+ G[u][v]
									* (1.0 - Y_in[u])
									* Y_in[u]
									* currentInput[j].value;
						for (j = numInputs; j < numInpHidCells; j++)
							SI[u][v][j] = SI[u][v][j] + G[u][v]
									* (1.0 - Y_in[u]) * Y_in[u] * Yk_mod_old[j];
					}
					/** weights to forget gate */
					if (forgetGates) {
						for (j = 0; j < currentInput.length; j++)
							SF[u][v][currentInput[j].index] = SF[u][v][currentInput[j].index]
									+ SC[u][v][currentInput[j].index]
									* (1.0 - Y_forget[u])
									* Y_forget[u]
									* currentInput[j].value;
						for (j = numInputs; j < numInpHidCells; j++)
							SF[u][v][j] = SF[u][v][j] + SC[u][v][j]
									* (1.0 - Y_forget[u]) * Y_forget[u]
									* Yk_mod_old[j];
					}
					/** weights to cell input */
					double input = inGates ? Y_in[u] : 1.0;

					for (j = 0; j < currentInput.length; j++)
						SC[u][v][currentInput[j].index] = SC[u][v][currentInput[j].index]
								+ input
								* (0.25 * (2.0 - G[u][v]) * (2.0 + G[u][v]))
								* currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++)
						SC[u][v][j] = SC[u][v][j] + input
								* (0.25 * (2.0 - G[u][v]) * (2.0 + G[u][v]))
								* Yk_mod_old[j];
				}
		}
	}

	@Override
	protected void backwardPass() {
		int k, i, j, u, v;
		double sum;
		double e[] = new double[numAll];
		double ec[][] = new double[numBlocks][cellblockSize];
		double eo[] = new double[numBlocks];
		double es[][] = new double[numBlocks][cellblockSize];

		/** output units */

		for (k = numInpHidCells, j = 0; k < numAll; k++, j++) {
			if (softmax)
				e[k] = error[j];
			else
				e[k] = error[j] * (1.0 - Yk_mod_new[k]) * Yk_mod_new[k];
			/** weight update contribution */
			for (i = numInputs; i < numHiddenAndInput; i++) {
				DW[k][i] += alpha * e[k] * Yk_mod_new[i];
			}

			i = numHiddenAndInput - 1;
			for (u = 0; u < numBlocks; u++) {
				if (inGates)
					i++;
				if (forgetGates)
					i++;
				if (outGates)
					i++;
				for (v = 0; v < blockSize[u]; v++) {
					i++;
					if(!dropout || dropouts[u])
						DW[k][i] += alpha * e[k] * Yk_mod_new[i];
				}
			}

		}

		/** hidden units */
		for (i = numInputs; i < numHiddenAndInput; i++) {
			sum = 0;
			for (k = numInpHidCells; k < numAll; k++)
				sum += W_mod[k][i] * e[k];
			e[i] = sum * (1.0 - Yk_mod_new[i]) * Yk_mod_new[i];
			/** weight update contribution */
			for(j = 0; j < currentInput.length; j++)
				DW[i][currentInput[j].index] += alpha * e[i] * currentInput[j].value;
			for (j = numInputs; j < numInpHidCells; j++) {
				DW[i][j] += alpha * e[i] * Yk_mod_old[j];
			}
		}

		/** error to memory cells ec[][] and internal states es[][] */
		i = numHiddenAndInput - 1;
		for (u = 0; u < numBlocks; u++) {
			if (inGates)
				i++;
			if (forgetGates)
				i++;
			if (outGates)
				i++;
			for (v = 0; v < blockSize[u]; v++) {
				i++;
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (k = numInpHidCells; k < numAll; k++)
						sum += W_mod[k][i] * e[k];
					ec[u][v] = sum;
					if (!outGates)
						es[u][v] = (0.5 * (1.0 + H[u][v]) * (1.0 - H[u][v])) * sum;
					else
						es[u][v] = Y_out[u]
								* (0.5 * (1.0 + H[u][v]) * (1.0 - H[u][v])) * sum;
				}
			}
		}

		/** output gates */
		if (outGates)
			for (u = 0; u < numBlocks; u++) {
				if(!dropout || dropouts[u]) {
					sum = 0;
					for (v = 0; v < blockSize[u]; v++) {
						sum += H[u][v] * ec[u][v];
					}
					eo[u] = sum * (1.0 - Y_out[u]) * Y_out[u];
				}
			}

		/** Derivatives of the internal state */
		derivatives();

		/** updates for weights to input and output gates and memory cells */
		i = numHiddenAndInput - 1;
		for (u = 0; u < numBlocks; u++) {
			/** input gate */
			if (inGates) {
				i++;
				if (!dropout || dropouts[u]) {
					for (j = 0; j < numInpHidCells; j++) {
						sum = 0;
						for (v = 0; v < blockSize[u]; v++) {
							sum += es[u][v] * SI[u][v][j];
						}
						DW[i][j] += alpha * sum;
					}
				}
			}
			/** output gate */
			if (outGates) {
				i++;
				if (!dropout || dropouts[u]){
					for(j = 0; j < currentInput.length; j++)
						DW[i][currentInput[j].index] += alpha * eo[u] * currentInput[j].value;
					for (j = numInputs; j < numInpHidCells; j++) {
						DW[i][j] += alpha * eo[u] * Yk_mod_old[j];
					}
				}
			}
			if (forgetGates) {
				i++;
				/** forget gate */
				if (!dropout || dropouts[u]) {
					for (j = 0; j < numInpHidCells; j++) {
						sum = 0;
						for (v = 0; v < blockSize[u]; v++) {
							sum += es[u][v] * SF[u][v][j];
						}
						DW[i][j] += alpha * sum;
					}
				}
			}
			/** memory cells */
			for (v = 0; v < blockSize[u]; v++) {
				i++;
				if (!dropout || dropouts[u])
					for (j = 0; j < numInpHidCells; j++)
						DW[i][j] += alpha * es[u][v] * SC[u][v][j];
			}
			
		}
		double lambda = parent.getDoubleParameter("lambda");
		if(lambda != 0)
			for(i = 0; i < DW.length; i++)
				for(j = 0; j < DW[i].length; j++)
					DW[i][j] += lambda * parentCore.W_mod[i][j];
	}

	public void train() {
		this.setParameters();
		this.initNet();
		validationEpochs = new LinkedHashMap<Integer, Double>();
		initialize();

		/** current validation set error minimum during training. */
		double currentMin = Double.POSITIVE_INFINITY;
		/**
		 * number of epochs since the current minimum of the validation error
		 * has been reached.
		 */
		int stepsSinceMin = 0;
		
		List<SparseLSTMCore<E>> processors = null;
		if(this.trainInParallel){
			processors = new ArrayList<SparseLSTMCore<E>>();
			for(int i = 0; i < Runtime.getRuntime().availableProcessors(); i++){
				SparseLSTMCore<E> lstm = new SparseLSTMCore<E>(baseDir, baseDir, parent, i * 23, outGates, forgetGates, inGates, dropout);		
				lstm.softmax = softmax;
				lstm.input = input;
				lstm.target = target;
				lstm.targetTest = targetTest;
				lstm.inputTest = inputTest;
				lstm.setParameters();
				lstm.initNet();
				lstm.W_mod = W_mod;
				lstm.parentCore = this;
				lstm.DW = DW;
				processors.add(lstm);
			}
		}

		for (epoch = 0; epoch < maxepoch; epoch++) {
			/** do a weight update or not. */
			boolean weightUpdate = false;
			
			/** MSE and misclassifications output for training set */
			outputEpoch();
			/** resetting the error per epoch */
			epochErr = 0;

			/** weight update after each epoch? */
			if (!weightUpdatePerSequence)
				weightUpdate = true;
			/** performing a test on a test set? */
			if ((epoch % validationFrequency == 0)
					&& (epoch > 0 || !randomWeightInitialization)) {
				resetNet();
				testLSTM();
				validationEpochs.put(epoch, Math.sqrt(epochErr / numForecasts));
				Gnuplot.plotOneDimensionalCurve(validationEpochs, "Epochs",
						this.baseDir + "validationEpochs");
				if (currentMin > epochErr) {
					currentMin = epochErr;
					stepsSinceMin = 0;
					save(baseDir + "maxnet.txt");
				} else {
					stepsSinceMin++;
				}
				if (stepsSinceMin > parent
						.getIntParameter("maxEpochsAfterMax"))
					break;
				numForecasts = 0;
			}
			double alphaOld = alpha;
			if (!this.trainInParallel)
				processExamples(weightUpdate, alphaOld, 0, input.length);
			else {
				int[][] startEnd = new int[processors.size()][2];
				int num = (int) ((double) input.length / startEnd.length);
				int lastEnd = 0;
				for (int i = 0; i < processors.size(); i++) {
					startEnd[i][0] = lastEnd;
					startEnd[i][1] = startEnd[i][0] + num;
					lastEnd = startEnd[i][1];
				}
				startEnd[startEnd.length - 1][1] = input.length;
				
				
				ExecutorService service = Executors.newFixedThreadPool(Runtime
						.getRuntime().availableProcessors());
				
				for (int i = 0; i < processors.size(); i++){
					processors.get(i).setSubThreadParameters(weightUpdate, alphaOld,
							startEnd[i][0], startEnd[i][1]);
					service.submit(processors.get(i));
				}
					//processors.get(i).processExamples(weightUpdate, alphaOld,
							//startEnd[i][0], startEnd[i][1]);
				
				service.shutdown();
				try {
					service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
				} catch (InterruptedException e) {
					e.printStackTrace();
					System.exit(-1);
				}
			}
		}

		this.minimumValidationError = currentMin;
		
		this.load(baseDir + "maxnet.txt");
		/** release memory. */
		this.input = null;
		this.target = null;
		this.inputTest = null;
		this.targetTest = null;
	}
	
	@Override
	protected void processExamples(boolean weightUpdate, double alphaOld, int start, int end) {
		for (int example = start; example < end; example++) {
			alpha = alphaOld;
			if(dropout){
				dropouts = new boolean[numBlocks];
				for(int i = 0; i < dropouts.length; i++)
					dropouts[i] = this.rand.nextBoolean();
			}
			resetNet();
			this.adjustAlpha(example);
			parentCore.numbSeq++;
			if (weightUpdatePerSequence || example % batchSize  == 0)
				weightUpdate = true;
			for (int element = 0; element < input[example].length; element++) {
				setInput(input[example][element],
						target[example][element]);

				forwardPass(false);

				if (targetExists) /** only if target for this input */
				{
					/** compute error */
					for (int k = numInpHidCells, j = 0; k < numAll; k++, j++) {
						error[j] = target_a[j] - Yk_mod_new[k];
					}

					/** Training error */
					parentCore.epochErr += compError(example, false);
				}

				/** backward pass */
				if (targetExists) /* only if target for this input */
					backwardPass();
				else
					derivatives();

				/** set old activations */
				for (int i = 0; i < numAll; i++) {
					Yk_mod_old[i] = Yk_mod_new[i];
				}

				/** update weights */
				if (weightUpdate) {
					weightUpdate = false;
					parentCore.weightUpdate();
				}
			}
		}
	}

	public void test(String netFile, DataSet<E> testData) {
		this.setParameters();
		this.initNet();
		if (netFile != null)
			this.load(this.baseDir + netFile);

		int i, k, j;
		epochErr = 0;
		numForecasts = 0;
		
		if(dropout){
			dropouts = new boolean[this.numBlocks];
			for(i = 0; i < dropouts.length; i++)
				dropouts[i] = true;
			multiplyMemoryCellWeights(0.5);
		}

		for (int currentSequence = 0; currentSequence < inputTest.length; currentSequence++) {
			resetNet();
			Sequence seq = testData.get(currentSequence);
			int gtSize = seq.getGroundTruthLength();
			
			if (seq.hasGroundTruthSequence() &&  gtSize == seq.getSequenceLength() && gtSize > 0)
				seq.initSequenceResults();	
			
			for (int currentElement = 0; currentElement < inputTest[currentSequence].length; currentElement++) {
				setInput(inputTest[currentSequence][currentElement],
						targetTest[currentSequence][currentElement]);
				forwardPass(true);

				if (targetExists) /** only if target for this input */
				{
					/** compute error */
					for (k = numInpHidCells, j = 0; k < numAll; k++, j++) {
						seq.addSequenceResult(currentElement, j,
								Yk_mod_new[k]);
						error[j] = target_a[j] - Yk_mod_new[k];
						parent.setResults(seq, j, Yk_mod_new[k], threadName);
					}

					/** Training error */
					epochErr += compError(currentSequence, true);
				}

				/** set old activations */
				for (i = 0; i < numAll; i++) {
					Yk_mod_old[i] = Yk_mod_new[i];
				}
			}
		}
		
		if(dropout)
			multiplyMemoryCellWeights(2.0);

		Log.printf("[LSTM-Thread: %s] TEST: epochs:%d sequences:%d\n",
				threadName, epoch + 1, numbSeq);
		Log.printf("[LSTM-Thread: %s] TEST: " + (this.softmax ? "cross entropy" : "RMSE") + ":%.6f\n", threadName,
				Math.sqrt(epochErr / (1.0 * numForecasts)));
		Log.printf("\n");

		this.inputTest = null;
		this.targetTest = null;

	}

	public void setTrainingData(Entry[][][] input, double[][][] target) {
		this.input = input;
		this.target = target;
	}

	public void setTestData(Entry[][][] input, double[][][] target) {
		this.inputTest = input;
		this.targetTest = target;
	}

}
