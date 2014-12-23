package ch.eonum.pipeline.classification.lstm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.util.FileUtil;

/**
 * Long Short Term Memory Recurrent Neural Network for regression.
 * http://en.wikipedia.org/wiki/Long_short_term_memory
 * 
 * Including n-fold cross validation.
 * 
 * @author tim
 *
 */
public class LSTM<E extends Sequence> extends Classifier<E> {
	
	/** all LSTM nets being used for this classifier. */
	protected List<LSTMCore<E>> nets;
	/**
	 * matrix for storing the target per training element per sequence element
	 * per output component
	 */
	protected double target[][][];
	/**
	 * matrix for storing the input per training element per sequence element per
	 * input component
	 */
	protected double input[][][];
	/**
	 * matrix for storing the target per test element per sequence element per
	 * output component
	 */
	protected double targetTest[][][];
	/**
	 * matrix for storing the input per test element per sequence element per
	 * input component
	 */
	protected double inputTest[][][];
	/** do always use the test set for epoch validation, even when using n-fold cross validation. */
	protected boolean alwaysUseValidationSet;
	/** maximum target value. 1.0 if norm is false. */
	protected double maxOutcome;
	/** norm all targets to maxOutcome. */
	protected boolean norm;
	/** use output gates. */
	protected boolean outputGates;
	/** use forget gates. */
	protected boolean forgetGates;
	/** use input gates. */
	protected boolean inputGates;
	/** do dropout. */
	protected boolean dropout;
	/** train in parallel on the core level. */
	protected boolean trainInParallel;
	
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("learningRate", "learning rate > 0 (default 0.01)");
		PARAMETERS.put("numLSTM", "number of memory cell blocks (default 2.0)");
		PARAMETERS.put("memoryCellBlockSize", "memory cell block sizes (default 2.0)");
		PARAMETERS.put("outputBias", "output layer bias: 1 --> biased 0 --> not biased (default 1.0)");
		PARAMETERS.put("hiddenBias", "hidden layer bias: 1 --> biased 0 --> not biased (default 1.0)");
		PARAMETERS.put("numHidden", "number of conventional hidden units (default 0.0)");
		PARAMETERS.put("initRange", "setting this value to a means that " +
				" the interval for weight initialization is [-a,a] (default 0.1)");
		PARAMETERS.put("gaussRange", "gauss range for weight initialization [0, INF] (default 1.0)");
		PARAMETERS.put("maxEpochs", "maximum number of epochs (default 1000000)");
		PARAMETERS.put("maxEpochsAfterMax", "maximum number of epochs after having reached" +
				"the minimum MSE on the Training set(default 20)");
		PARAMETERS.put("numNets", "number of nets which are to be trained per net(default: 1.0)");
		PARAMETERS.put("numNetsTotal", "number of nets which are to be trained" +
				" to build a network of nets (default: 1.0)");
		PARAMETERS.put("normTarget", "norm all targets by dividing by normTarget." +
				" If set to -1.0, maxOutcome is used for norming (default: -1.0)");
		PARAMETERS.put("lambda", "lambda for weight decay. Default: 0.0 (no weight decay)");
		PARAMETERS.put("batchSize", "batch size for mini-batch weight updates Default: 1.0 (online learning)");
		PARAMETERS.put("momentum",  "momentum for weigth updates in backpropagation : 0.0 (no momentum)");
	}

	public LSTM() {
		this.setSupportedParameters(LSTM.PARAMETERS);
		this.putParameter("learningRate", 0.01);
		this.putParameter("numLSTM", 2.0);
		this.putParameter("memoryCellBlockSize", 2.0);
		this.putParameter("outputBias", 1.0);
		this.putParameter("hiddenBias", 1.0);
		this.putParameter("numHidden", 0.0);
		this.putParameter("initRange", 0.1);
		this.putParameter("gaussRange", 1.0);
		this.putParameter("maxEpochs", 1000000);
		this.putParameter("maxEpochsAfterMax", 20);
		this.putParameter("numNets", 1.0);
		this.putParameter("numNetsTotal", 1.0);
		this.putParameter("normTarget", -1.0);
		this.putParameter("lambda", 0.0);
		this.putParameter("momentum", 0.0);
		this.putParameter("batchSize", 1.0);
		alwaysUseValidationSet = false;
		norm = false;
		forgetGates = true;
		outputGates = true;
		inputGates = true;
		dropout = false;
		classify = false;
	}

	@Override
	public void train() {
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		int numNets = (int)this.getDoubleParameter("numNets");
		
		this.maxOutcome = Double.NEGATIVE_INFINITY;
		for(Instance each : trainingDataSet)
			maxOutcome = Math.max(maxOutcome, each.outcome);
		if(!norm)
			maxOutcome = 1.0;
		else if(this.getDoubleParameter("normTarget") > 0.0)
			maxOutcome = getDoubleParameter("normTarget");
		
		double[][][][] trainInputs = null;
		double[][][][] trainTargets = null;
		if(numNetsTotal > 1){
			trainInputs = new double[numNetsTotal][][][];
			trainTargets = new double[numNetsTotal][][][];
			int i = 0;
			for (DataSet<E> each : this.trainingDataSet.splitIntoNSubsets(numNetsTotal)) {
				trainInputs[i] = this.loadDataSet(each);
				trainTargets[i] = this.loadTargets(each);
				i++;
			}
			if(this.alwaysUseValidationSet){
				this.inputTest = this.loadDataSet(this.testDataSet);
				this.targetTest = this.loadTargets(this.testDataSet);
			}
		} else {
			this.input = this.loadDataSet(this.trainingDataSet);
			this.target = this.loadTargets(this.trainingDataSet);
			this.inputTest = this.loadDataSet(this.testDataSet);
			this.targetTest = this.loadTargets(this.testDataSet);
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Runtime
				.getRuntime().availableProcessors());
		
		ArrayList<List<LSTMCore<E>>> allNets = new ArrayList<List<LSTMCore<E>>>();
		
		for (int netNumber = 0; netNumber < numNetsTotal; netNumber++) {
			if (numNetsTotal > 1) {
				if (!this.alwaysUseValidationSet){
					inputTest = trainInputs[netNumber];
					targetTest = trainTargets[netNumber];
				}
				int length = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if(i != netNumber)
						length += trainInputs[i].length;
				input = new double[length][][];
				target = new double[length][][];
				int k = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if (i != netNumber)
						for(int j = 0; j < trainInputs[i].length; j++){
							input[k] = trainInputs[i][j];
							target[k++] = trainTargets[i][j];
						}
			}
			
			ArrayList<LSTMCore<E>> randomNets = new ArrayList<LSTMCore<E>>();
			for(int i = 0; i < numNets; i++){
				FileUtil.mkdir(this.getBaseDir() + netNumber + "-" + i + "/");
				randomNets.add(createNet(netNumber + "-" + i, this.getBaseDir() + netNumber + "-" + i + "/", i * 11));
			}

			for (int numNet = 0; numNet < numNets; numNet++){
				randomNets.get(numNet).setTrainingData(input, target);
				randomNets.get(numNet).setTestData(inputTest, targetTest);
			}
			
			for (int numNet = 0; numNet < numNets; numNet++){
				service.submit(randomNets.get(numNet));
			}
			
			allNets.add(randomNets);
			
		}
		
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		this.nets = new ArrayList<LSTMCore<E>>();
		for (int netNumber = 0; netNumber < numNetsTotal; netNumber++) {
			double max = Double.POSITIVE_INFINITY;
			int maxNet = -1;
			List<LSTMCore<E>> randomNets = allNets.get(netNumber);
			for (int numNet = 0; numNet < numNets; numNet++){
				if(randomNets.get(numNet).getValidationError() < max){
					max = randomNets.get(numNet).getValidationError();
					maxNet = numNet;
				}
			}
			
			this.refineNets(maxNet);
			randomNets.get(maxNet).save(this.getBaseDir() + "maxNNglobal" + netNumber);
			nets.add(randomNets.get(maxNet));
		}
		
		/** release memory. */
		this.input = null;
		this.target = null;
		this.inputTest = null;
		this.targetTest = null;
	}
	
	protected LSTMCore<E> createNet(String name, String folder, int seed) {
		LSTMCore<E> l = new LSTMCore<E>(name, folder, this, seed, outputGates, forgetGates, inputGates, dropout);
		if(this.trainInParallel)
			l.trainInParallel();
		return l ;
	}

	protected void refineNets(int maxNet) {
		// do nothing
	}

	protected double[][][] loadTargets(DataSet<E> set) {
		double data[][][] = new double[set.size()][][];
		for(int i = 0; i < set.size(); i ++){
			Sequence seq = set.get(i);
			int length = Math.max(1, seq.getSequenceLength());
			
			int outSize = seq.outputSize();
			
			if(seq.hasSequenceTarget()){
				data[i] = new double[length][outSize];
				for(int j = 0; j < length; j++)
					for(int k = 0; k < outSize; k++)
						data[i][j][k] = seq.groundTruthAt(j, k);
			} else {
				data[i] = new double[length][1];
				for(int j = 0; j < length; j++)
					for(int k = 0; k < data[i][j].length; k++)
						data[i][j][k] = Double.NaN;
				data[i][length-1][0] = seq.outcome / this.maxOutcome;
			}	
		}
		return data;
	}

	private double[][][] loadDataSet(DataSet<E> set) {
		double data[][][] = new double[set.size()][][];
		for(int i = 0; i < set.size(); i++)
			data[i] = set.get(i).getDenseRepresentation(features);
		return data;
	}

	@Override
	public DataSet<E> test() {
		
		this.inputTest = this.loadDataSet(this.testDataSet);
		this.targetTest = this.loadTargets(this.testDataSet);
		
		for(Instance each : this.testDataSet)
			each.putResult("result", 0.0);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			LSTMCore<E> net = createNet("Test" + netNumber, this.getBaseDir(), 0);
			net.setTestData(inputTest, targetTest);
			net.test("maxNNglobal" + netNumber, this.testDataSet);
			net.destroy();
		}
		
		this.fillResults(numNetsTotal);
		
		// release memory
		this.inputTest = null;
		this.targetTest = null;
		return this.testDataSet;
	}

	/**
	 * store results. normalize results.
	 * @param numNetsTotal
	 */
	protected void fillResults(int numNetsTotal) {
		for(Instance each : this.testDataSet)
			each.putResult("result", (each.getResult("result") / numNetsTotal) * this.maxOutcome);
	}
	
	/**
	 * set the result of one net for a sequence
	 * @param seq
	 * @param outputUnit #TODO use this
	 * @param activation
	 */
	public synchronized void setResults(Sequence seq, int outputUnit,
			double activation, String threadName) {
		seq.putResult("result", seq.getResult("result") + activation);
		seq.putResult("result" + threadName,
				activation * getMaxOutcome());
	}

	/**
	 * Do always use the test set for epoch validation.
	 * Even for n-fold cross validation.
	 */
	public void alwaysUseValidationSet() {
		this.alwaysUseValidationSet = true;
	}
	
	/**
	 * enable the norming of targets to [0,1]
	 */
	public void enableTargetNorming(){
		norm = true;
	}
	
	public void setOutputGateUse(boolean b){
		outputGates = b;
	}
	
	public void setForgetGateUse(boolean b){
		forgetGates = b;
	}
	
	public void setInputGateUse(boolean b){
		inputGates = b;
	}

	public double getMaxOutcome() {
		return this.maxOutcome;
	}

	public void doDropout() {
		this.dropout = true;
	}

	/**
	 * train using all processors even if you have only one net. Training in
	 * parallel on the core level is not deterministic yet!!
	 */
	public void trainInParallel() {
		this.trainInParallel = true;
	}
	
}