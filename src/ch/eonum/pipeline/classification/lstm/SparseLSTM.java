/**
 * 
 */
package ch.eonum.pipeline.classification.lstm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Entry;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseSequence;
import ch.eonum.pipeline.util.FileUtil;

/**
 * Sparse implementation of the LSTM network for regression.
 * 
 * @author tim
 *
 */
public class SparseLSTM<E extends SparseSequence> extends LSTM<E> {
	
	/** all LSTM nets being used for this classifier. */
	protected List<SparseLSTMCore<E>> nets;
	/**
	 * matrix for storing the input per training element per sequence element per
	 * input component
	 */
	protected Entry input[][][];
	/**
	 * matrix for storing the input per test element per sequence element per
	 * input component
	 */
	protected Entry inputTest[][][];
	

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
		
		Entry[][][][] trainInputs = null;
		double[][][][] trainTargets = null;
		if(numNetsTotal > 1){
			trainInputs = new Entry[numNetsTotal][][][];
			trainTargets = new double[numNetsTotal][][][];
			int i = 0;
			for (DataSet<E> each : this.trainingDataSet.splitIntoNSubsets(numNetsTotal)) {
				trainInputs[i] = this.loadDataSetSparse(each);
				trainTargets[i] = this.loadTargets(each);
				i++;
			}
			if(this.alwaysUseValidationSet){
				this.inputTest = this.loadDataSetSparse(this.testDataSet);
				this.targetTest = this.loadTargets(this.testDataSet);
			}
		} else {
			this.input = this.loadDataSetSparse(this.trainingDataSet);
			this.target = this.loadTargets(this.trainingDataSet);
			this.inputTest = this.loadDataSetSparse(this.testDataSet);
			this.targetTest = this.loadTargets(this.testDataSet);
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Runtime
				.getRuntime().availableProcessors());
		
		ArrayList<List<SparseLSTMCore<E>>> allNets = new ArrayList<List<SparseLSTMCore<E>>>();
		
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
				input = new Entry[length][][];
				target = new double[length][][];
				int k = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if (i != netNumber)
						for(int j = 0; j < trainInputs[i].length; j++){
							input[k] = trainInputs[i][j];
							target[k++] = trainTargets[i][j];
						}
			}
			
			ArrayList<SparseLSTMCore<E>> randomNets = new ArrayList<SparseLSTMCore<E>>();
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
		
		this.nets = new ArrayList<SparseLSTMCore<E>>();
		for (int netNumber = 0; netNumber < numNetsTotal; netNumber++) {
			double max = Double.POSITIVE_INFINITY;
			int maxNet = -1;
			List<SparseLSTMCore<E>> randomNets = allNets.get(netNumber);
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
	
	

	protected SparseLSTMCore<E> createNet(String name, String folder, int seed) {
		SparseLSTMCore<E> l = new SparseLSTMCore<E>(name, folder, this, seed, outputGates, forgetGates, inputGates, dropout);
		if(this.trainInParallel)
			l.trainInParallel();
		return l ;
	}

	private Entry[][][] loadDataSetSparse(DataSet<E> set) {
		Entry data[][][] = new Entry[set.size()][][];
		boolean biasHidden = this.getDoubleParameter("hiddenBias") > 0.5;
		for(int i = 0; i < set.size(); i++){
			data[i] = set.get(i).getSparseRepresentation(features, biasHidden);
		}
		return data;
	}

	@Override
	public DataSet<E> test() {
		
		this.inputTest = this.loadDataSetSparse(this.testDataSet);
		this.targetTest = this.loadTargets(this.testDataSet);
		
		for(Instance each : this.testDataSet)
			each.putResult("result", 0.0);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			SparseLSTMCore<E> net = createNet("Test" + netNumber, this.getBaseDir(), 0);
			net.setTestData(inputTest, targetTest);
			net.test("maxNNglobal" + netNumber, this.testDataSet);
			net.destroy();
		}
		
		this.fillResults(numNetsTotal);
		
		/** release memory. */
		this.inputTest = null;
		this.targetTest = null;
		return this.testDataSet;
	}
	
}
