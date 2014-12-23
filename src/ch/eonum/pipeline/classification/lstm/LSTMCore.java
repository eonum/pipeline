package ch.eonum.pipeline.classification.lstm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Core functionality (all the math) of the Long Short Term Memory Recurrent
 * Neural Network. Can be run as a thread for the training.
 * 
 * @author tim
 * 
 */
public class LSTMCore<E extends Sequence> implements Runnable {
	/** random number generator with seed. */
	protected Random rand;
	/** Thread Name/Number. */
	protected String threadName;
	/** base directory, where we store the net. */
	protected String baseDir;
	/** number of inputs/features */
	protected int numInputs;
	/** number of outputs/targets */
	protected int numTargets;
	/** number of hidden units */
	protected int numHidden;
	/** number input and hidden units */
	protected int numHiddenAndInput;
	/** number input and hidden units and gates and memory cells */
	protected int numInpHidCells;
	/** number of memory cell blocks */
	protected int numBlocks;
	/** Size of the memory cell blocks */
	protected int blockSize[];
	/** number of all units */
	protected int numAll;
	/** max. number of epochs before learning stop */
	protected int maxepoch;
	/** biasHidden --> hidden units, with gates, and cells biased */
	protected boolean biasHidden;
	/** biasOut --> output units biased */
	protected boolean biasOut;
	/** is target provided for the current input */
	protected boolean targetExists;
	/** do weight update per sequence otherwise per epoch. */
	protected boolean weightUpdatePerSequence;
	/** number of sequences seen by the net */
	protected int numbSeq;
	/** number of epochs seen by the net */
	protected int epoch;
	/** weight matrix */
	protected double W_mod[][];
	/** contribution to update of weight matrix */
	protected double DW[][];
	/** input gates */
	protected double Y_in[];
	/** output gates */
	protected double Y_out[];
	/** forget gates */
	protected double Y_forget[];
	/** new activation for all units */
	protected double Yk_mod_new[];
	/** old activation for all units */
	protected double Yk_mod_old[];
	/** function g for each cell */
	protected double G[][];
	/** function h for each cell */
	protected double H[][];
	/** internal state for each cell */
	protected double S[][];
	/** output for each cell */
	protected double Yc[][];
	/** derivative with respect to weights to input gate for each cell */
	protected double SI[][][];
	/** derivative with respect to weights to forget gate for each cell */
	protected double SF[][][];
	/** derivative with respect to weights to cell input for each cell */
	protected double SC[][][];
	/** initial input gate bias for each block */
	protected double biasInp[];
	/** initial output gate bias for each block */
	protected double biasOutput[];
	/** initial forget gate bias for each block */
	protected double biasForget[];
	/** learning rate */
	protected double alpha;
	/** interval of weight initialization is init_range*[-1,1] */
	protected double initRange;
	/** current target */
	protected double target_a[];
	/**
	 * matrix for storing the target per training element per sequence element
	 * per output component
	 */
	protected double target[][][];
	/**
	 * matrix for storing the input per training element per sequence element
	 * per input component
	 */
	private double input[][][];
	/**
	 * matrix for storing the target per test element per sequence element per
	 * output component
	 */
	protected double targetTest[][][];
	/**
	 * matrix for storing the input per test element per sequence element per
	 * input component
	 */
	private double inputTest[][][];
	/** MSE per epoch */
	protected double epochErr;
	/** error for output units */
	protected double error[];
	/** memory cell block size. */
	protected int cellblockSize;
	/** validation error for each epoch. */
	protected Map<Integer, Double> validationEpochs;
	/** parent LSTM object. all parameters can be obtained from there. */
	protected LSTM<E> parent;
	/**
	 * parent LSTM core object if this net is only a processor without it's own
	 * weights. if this is the central core, this is a pointer to this.
	 */
	protected LSTMCore<E> parentCore;
	/** train this net with multiple cores in parallel. */
	protected boolean trainInParallel;
	/**
	 * true if weights are to be initialized randomly. false if weights are set.
	 */
	protected boolean randomWeightInitialization;
	/** use forget gates. */
	protected boolean forgetGates;
	/** use output gates. */
	protected boolean outGates;
	/** use input gates. */
	protected boolean inGates;
	/** number of forecasts made during a test. used for calculation of RMSE. */
	protected int numForecasts;
	/** do a cross validation run each X epochs. */
	protected int validationFrequency;
	/** minimum validation error from cross validation. */
	protected double minimumValidationError;
	/** batch size. */
	protected int batchSize;
	/** do dropout? */
	protected boolean dropout;
	/** do a dropout on memory block i. */
	protected boolean[] dropouts = null;
	/** momentum for momentum learning. */
	protected double momentum;
	/** use the softmax activation function for the output layer. */
	protected boolean softmax;
	/** parameters for sub threads (parentCore != this). */
	protected boolean subThreadWeightUpdate;
	protected double subThreadAlphaOld;
	protected int subThreadStart;
	protected int subThreadStop;

	public LSTMCore(String threadName, String baseDir, LSTM<E> parent, int seed,
			boolean outputGates, boolean forgetGates, boolean inputGates, boolean dropout) {
		this.threadName = threadName;
		this.baseDir = baseDir;
		this.parent = parent;
		this.rand = new Random(seed);
		this.randomWeightInitialization = true;
		this.outGates = outputGates;
		this.forgetGates = forgetGates;
		this.inGates = inputGates;
		this.batchSize = 1;
		this.dropout = dropout;
		this.softmax = false;
		this.trainInParallel = false;
		this.parentCore = this;
	}

	/**
	 * allocate memory for the net.
	 */
	protected void initNet() {
		/** weight matrix */
		if (W_mod == null)
			W_mod = new double[numAll][numAll];
		/** contribution to update of weight matrix */
		DW = new double[numAll][numAll];
		/** input gates */
		if (inGates)
			Y_in = new double[numBlocks];
		/** output gates */
		if (outGates)
			Y_out = new double[numBlocks];
		/** forget gates */
		if (forgetGates)
			Y_forget = new double[numBlocks];
		/** new activation for all units */
		Yk_mod_new = new double[numAll];
		/** old activation for all units */
		Yk_mod_old = new double[numAll];
		/** function g for each cell */
		G = new double[numBlocks][cellblockSize];
		/** function h for each cell */
		H = new double[numBlocks][cellblockSize];
		/** internal state for each cell */
		S = new double[numBlocks][cellblockSize];
		/** output for each cell */
		Yc = new double[numBlocks][cellblockSize];
		/** derivative with respect to weights to input gate for each cell */
		if (inGates)
			SI = new double[numBlocks][cellblockSize][numAll];
		/** derivative with respect to weights to cell input for each cell */
		SC = new double[numBlocks][cellblockSize][numAll];
		/** derivative with respect to weights to forget gate for each cell */
		if (forgetGates)
			SF = new double[numBlocks][cellblockSize][numAll];
		/** current target */
		target_a = new double[numTargets];
		/** error for output units */
		error = new double[numTargets];
	}

	/**
	 * random number generation. Gauss distributed.
	 * @param k
	 * @return
	 */
	protected int seprand(int k) {
		return (int) ((1 / Math.sqrt(2 * Math.PI))
				* Math.exp(-Math.pow(
						rand.nextDouble()
								* parent.getDoubleParameter("gaussRange"), 2.0) / 2.0) * k);
	}

	/**
	 * reset the net's state.
	 */
	protected void resetNet() {
		int i, j, v, u;

		for (i = 0; i < numAll; i++) {
			Yk_mod_new[i] = dropout ? 0.0 : 0.5;
			Yk_mod_old[i] = dropout ? 0.0 : 0.5;
		}
		for (u = 0; u < numBlocks; u++) {
			for (v = 0; v < blockSize[u]; v++) {
				S[u][v] = 0;
				G[u][v] = 0;
				H[u][v] = 0;
				Yc[u][v] = dropout ? 0.0 : 0.5;
				for (j = 0; j < numInpHidCells; j++) {
					if (inGates)
						SI[u][v][j] = 0;
					SC[u][v][j] = 0;
					if (forgetGates)
						SF[u][v][j] = 0;
				}
			}
		}

	}

	protected void forwardPass(boolean testTime) {
		int i, j, u, v, k;
		double sum;

		/** ### hidden units ### */
		for (i = numInputs; i < numHiddenAndInput - (biasOut ? 1 : 0); i++) {
			sum = 0;
			for (j = 0; j < numInpHidCells; j++)
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
					for (j = 0; j < numInpHidCells; j++)
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
					for (j = 0; j < numInpHidCells; j++)
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
					for (j = 0; j < numInpHidCells; j++)
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
					for (j = 0; j < numInpHidCells; j++)
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

		/** output units activation */
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

	protected double compError(int currentSequence, boolean test) {
		int k, j;
		double err = 0;
		for (k = numInpHidCells, j = 0; k < numAll; k++, j++) {
			if(softmax)
				err -= this.target_a[j] * Math.log(Yk_mod_new[k]);
			else
				err += error[j] * error[j];
			parentCore.numForecasts++;
		}
		return err;
	}

	protected void outputEpoch() {
		Log.printf("[LSTM-Thread: %s] epochs:%d sequences:%d\n", threadName,
				epoch + 1, numbSeq);
		Log.printf("[LSTM-Thread: %s] " + (this.softmax ? "cross entropy" : "RMSE") + ":%.6f\n", threadName,
				Math.sqrt(epochErr / numForecasts));
		Log.printf("\n");
	}

	/**
	 * Test the net on the validation set.
	 */
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
				Math.sqrt(epochErr / numForecasts));

		Log.printf("\n");
	}

	/**
	 * multiply all weights within a memory cell including gates by factor.
	 * This is used during test time when doing dropout during training.
	 * 
	 * @param factor
	 */
	protected void multiplyMemoryCellWeights(double factor) {
		int i = numHiddenAndInput - 1;
		int j, v;
		for (int u = 0; u < numBlocks; u++) {

			/** input gate */
			if (inGates) {
				i++;
				for (j = 0; j < W_mod.length; j++)
					W_mod[j][i] *= factor;
			}

			/** output gate */
			if (outGates) {
				i++;
				for (j = 0; j < W_mod.length; j++)
					W_mod[j][i] *= factor;
			}

			/** forget gate */
			if (forgetGates) {
				i++;
				for (j = 0; j < W_mod.length; j++)
					W_mod[j][i] *= factor;
			}

			/** uth memory cell block */
			for (v = 0; v < blockSize[u]; v++) {
				i++;
				for (j = 0; j < W_mod.length; j++)
					W_mod[j][i] *= factor;
			}
		}
	}

	/**
	 * feed the net with data.
	 * @param input
	 * @param target
	 */
	private void setInput(double[] input, double[] target) {
		int i, j, k;

		for (i = 0; i < numInputs - (biasHidden ? 1 : 0); i++) {
			Yk_mod_new[i] = input[i];
			Yk_mod_old[i] = Yk_mod_new[i];
		}
		if (biasHidden) {
			Yk_mod_new[numInputs - 1] = 1.0;
			Yk_mod_old[numInputs - 1] = 1.0;
		}

		targetExists = true;
		for (k = numInpHidCells, j = 0; k < numAll; k++, j++) {
			target_a[j] = target[j];
			if (Double.isNaN(target[j]))
				targetExists = false;
		}			
	}

	/**
	 * set meta parameters and calculate architecture metrics.
	 */
	protected void setParameters() {
		this.numInputs = parent.getFeatures().size();
		this.numTargets = this.targetTest[0][0].length;

		this.numBlocks = (int) parent.getDoubleParameter("numLSTM");
		this.alpha = parent.getDoubleParameter("learningRate");
		this.momentum = parent.getDoubleParameter("momentum");
		this.biasOut = parent.getDoubleParameter("outputBias") > 0.5;
		this.numHidden = (int) parent.getDoubleParameter("numHidden");
		this.biasHidden = parent.getDoubleParameter("hiddenBias") > 0.5;
		this.batchSize = (int) parent.getDoubleParameter("batchSize");

		cellblockSize = (int) parent.getDoubleParameter("memoryCellBlockSize");

		blockSize = new int[numBlocks];
		if (inGates)
			biasInp = new double[numBlocks];
		if (outGates)
			biasOutput = new double[numBlocks];
		if (forgetGates)
			biasForget = new double[numBlocks];

		for (int i = 0; i < numBlocks; i++)
			blockSize[i] = cellblockSize;
		if (inGates)
			for (int i = 0; i < numBlocks; i++)
				biasInp[i] = -i - 1;
		if (outGates)
			for (int i = 0; i < numBlocks; i++)
				biasOutput[i] = 0;
		if (forgetGates)
			for (int i = 0; i < numBlocks; i++)
				biasForget[i] = 0.5 * i + 1;

		this.initRange = parent.getDoubleParameter("initRange");
		this.validationFrequency = 1;
		
		this.weightUpdatePerSequence = batchSize <= 1;
		this.maxepoch = parent.getIntParameter("maxEpochs"); //

		if (biasHidden)
			numInputs++;
		if (biasOut)
			numHidden++;
		numHiddenAndInput = numInputs + numHidden;
		numInpHidCells = numHiddenAndInput;
		int numGates = 0;
		if (inGates)
			numGates++;
		if (forgetGates)
			numGates++;
		if (outGates)
			numGates++;
		for (int i = 0; i < numBlocks; i++)
			numInpHidCells += (numGates + blockSize[i]);
		numAll = numInpHidCells + numTargets;
	}

	protected void initialize() {
		int i, j, u, v;
		epoch = 0;
		epochErr = 0;
		numbSeq = 0;

		/** weight initialization */
		if (this.randomWeightInitialization) {
			for (i = 0; i < numAll; i++) {
				for (j = 0; j < numAll; j++) {
					W_mod[i][j] = (seprand(2000) - 1000);
					W_mod[i][j] /= 1000.0;
					W_mod[i][j] *= initRange;
					DW[i][j] = 0;
				}
			}

			/** gates bias initialization */
			if (biasHidden) {
				i = numHiddenAndInput - 1;
				for (u = 0; u < numBlocks; u++) {
					if (inGates) {
						i++;
						W_mod[i][numInputs - 1] += biasInp[u];
					}
					if (outGates) {
						i++;
						W_mod[i][numInputs - 1] += biasOutput[u];
					}
					if (forgetGates) {
						i++;
						W_mod[i][numInputs - 1] += biasForget[u];
					}
					for (v = 0; v < blockSize[u]; v++) {
						i++;
					}
				}
			}
		}

		/** reset activations and derivatives */
		resetNet();
	}

	/**
	 *  Derivatives of the internal state over time.
	 */
	protected void derivatives() {
		int u, v, j;
		for (u = 0; u < numBlocks; u++) {
			if(!dropout || dropouts[u])
				for (v = 0; v < blockSize[u]; v++) {
					/** weights to input gate */
					if (inGates)
						for (j = 0; j < numInpHidCells; j++)
							SI[u][v][j] = SI[u][v][j] + G[u][v] * (1.0 - Y_in[u])
									* Y_in[u] * Yk_mod_old[j];
					/** weights to forget gate */
					if (forgetGates)
						for (j = 0; j < numInpHidCells; j++)
							SF[u][v][j] = SF[u][v][j] + SC[u][v][j]
									* (1.0 - Y_forget[u]) * Y_forget[u]
									* Yk_mod_old[j];
					/** weights to cell input */
					double input = inGates ? Y_in[u] : 1.0;
					for (j = 0; j < numInpHidCells; j++)
						SC[u][v][j] = SC[u][v][j] + input
								* (0.25 * (2.0 - G[u][v]) * (2.0 + G[u][v]))
								* Yk_mod_old[j];
				}
		}
	}

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
			for (j = 0; j < numInpHidCells; j++) {
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
				if (!dropout || dropouts[u])
					for (j = 0; j < numInpHidCells; j++) {
						DW[i][j] += alpha * eo[u] * Yk_mod_old[j];
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

	protected synchronized void weightUpdate() {
		for(int i = 0; i < DW.length; i++)
			for(int j = 0; j < DW[i].length; j++){
				W_mod[i][j] += DW[i][j];
				DW[i][j] *= momentum ;
			}
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
		
		List<LSTMCore<E>> processors = null;
		if(this.trainInParallel){
			processors = new ArrayList<LSTMCore<E>>();
			for(int i = 0; i < Runtime.getRuntime().availableProcessors(); i++){
				LSTMCore<E> lstm = new LSTMCore<E>(baseDir, baseDir, parent, i * 23, outGates, forgetGates, inGates, dropout);		
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

	protected void setSubThreadParameters(boolean weightUpdate, double alphaOld,
			int start, int stop) {
		this.subThreadWeightUpdate = weightUpdate;
		this.subThreadAlphaOld = alphaOld;
		this.subThreadStart = start;
		this.subThreadStop = stop;
	}

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

	protected void adjustAlpha(int example) {
		/** do nothing. */
	}

	public double getValidationError() {
		return this.minimumValidationError;
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

	public void setTrainingData(double[][][] input, double[][][] target) {
		this.input = input;
		this.target = target;
	}

	public void setTestData(double[][][] input, double[][][] target) {
		this.inputTest = input;
		this.targetTest = target;
	}

	@Override
	public void run() {
		try {
			if(parentCore == this)
				this.train();
			else
				this.processExamples(subThreadWeightUpdate, subThreadAlphaOld, subThreadStart, subThreadStop);
		} catch (Exception e){
			Log.puts("Exception in Thread " + this.threadName + " " + e.getMessage());
			e.printStackTrace();
			System.exit(-1);
		}
	}

	/**
	 * get the weight matrix.
	 * @return
	 */
	public double[][] getWeightMatrix() {
		return W_mod;
	}

	/**
	 * set a precomputed / initialized weight matrix.
	 * @param w_mod
	 */
	public void setWeightMatrix(double[][] w_mod) {
		this.randomWeightInitialization = false;
		W_mod = w_mod;
	}

	/**
	 * release all memory except the net itself.
	 */
	public void destroy() {
		this.parent = null;
		this.DW = null;
		this.input = null;
		this.inputTest = null;
		this.target = null;
		this.targetTest = null;
	}

	/**
	 * save the weight matrix to a file.
	 * @param fileName
	 */
	public void save(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			int i, j;
			for (i = 0; i < numAll; i++) {
				for (j = 0; j < numAll; j++)
					p.printf("(%d,%d): %.6f ", i, j, W_mod[i][j]);
				p.printf("\n");
			}
			p.printf("\n");
			p.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * load the weight matrix from a file.
	 * @param filename
	 */
	public void load(String filename) {
		try {
			FileInputStream fstream = new FileInputStream(filename);
			BufferedReader br = new BufferedReader(new InputStreamReader(
					fstream));
			String line;
			this.W_mod = new double[numAll][numAll];
			for (int i = 0; i < numAll; i++) {
				line = br.readLine();
				StringTokenizer st = new StringTokenizer(line);
				for (int j = 0; j < numAll; j++) {
					st.nextToken();
					W_mod[i][j] = Double.parseDouble(st.nextToken());
				}
			}
			br.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * softmax output layer for classification and probability estimation tasks.
	 */
	public void useSoftMaxOutputLayer() {
		this.softmax = true;
	}
	
	/**
	 * train using all processors. Training in parallel on the core level is not
	 * deterministic yet!!
	 */
	public void trainInParallel(){
		this.trainInParallel = true;
	}

}
