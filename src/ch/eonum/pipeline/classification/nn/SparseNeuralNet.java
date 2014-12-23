package ch.eonum.pipeline.classification.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Entry;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Log;

/**
 * Same functionality as @see NeuralNet but with sparse representations of input
 * data and input layer. For sparse data this net's performance is much better
 * than that of the dense NeuralNetwork.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class SparseNeuralNet<E extends SparseInstance> extends NeuralNet<E> {

	private ArrayList<SparseNeuralNetCore<E>> netsE;

	public SparseNeuralNet(Features features) {
		super(features);
	}
	
	/**
	 * Get a data set as an Entry[][] matrix. Each row represents an instance.
	 * Each column represents a non zero feature.
	 * 
	 * @param features
	 * @return
	 */
	public Entry[][] asEntryArrayMatrix(DataSet<E> ds, Features features) {
		Entry[][] matrix = new Entry[ds.size()][];
		for (int i = 0; i < ds.size(); i++)
			matrix[i] = ds.get(i).asEntryArray(features);
		
		return matrix;
	}
	
	@Override
	public void train() {
		Log.puts("Sparse Neural Net: Start Training");
		if(classify)
			prepareClasses();
		
		this.maxOutcome = Double.NEGATIVE_INFINITY;
		for(Instance each : trainingDataSet)
			maxOutcome = Math.max(maxOutcome, each.outcome);
		if(this.getDoubleParameter("normTarget") > 0.0)
			maxOutcome = getDoubleParameter("normTarget");
		this.putParameter("maxOutcome", maxOutcome);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		int numNets = (int)this.getDoubleParameter("numNets");
		
		Entry[][][] trainInputs = null;
		Entry[][] trainTargets = null;
		
		Entry[][] input = null;
		Entry[] target = null;
		Entry[][] inputTest = null;
		Entry[] targetTest = null;
		
		if(numNetsTotal > 1){
			trainInputs = new Entry[numNetsTotal][][];
			trainTargets = new Entry[numNetsTotal][];
			int i = 0;
			List<DataSet<E>> sets = this.trainingDataSet.splitIntoNSubsets(numNetsTotal);
			for (DataSet<E> each : sets) {
				trainInputs[i] = asEntryArrayMatrix(each, features);
				trainTargets[i] = getTargets(each);
				i++;
			}
		} else {
			input = asEntryArrayMatrix(this.trainingDataSet, features);
			target = getTargets(this.trainingDataSet);
			inputTest = asEntryArrayMatrix(this.testDataSet, features);
			targetTest = getTargets(this.testDataSet);
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Runtime
				.getRuntime().availableProcessors());
		
		ArrayList<List<SparseNeuralNetCore<E>>> allNets = new ArrayList<List<SparseNeuralNetCore<E>>>();
		
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){			
			if (numNetsTotal > 1) {
				inputTest = trainInputs[netNumber];
				targetTest = trainTargets[netNumber];
				
				int length = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if(i != netNumber)
						length += trainInputs[i].length;
				input = new Entry[length][];
				target = new Entry[length];
				int k = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if (i != netNumber)
						for(int j = 0; j < trainInputs[i].length; j++){
							input[k] = trainInputs[i][j];
							target[k++] = trainTargets[i][j];
						}
			}
			
			ArrayList<SparseNeuralNetCore<E>> randomNets = new ArrayList<SparseNeuralNetCore<E>>();
			
			for(int i = 0; i < numNets; i++){
				String name = netNumber + "-" + i;
				FileUtil.mkdir(this.getBaseDir() + name + "/");
				SparseNeuralNetCore<E> net = createNet(name, i);
				net.setTrainingData(input, target);
				net.setTestData(inputTest, targetTest);
				randomNets.add(net);
				service.submit(net);
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
		
		this.netsE = new ArrayList<SparseNeuralNetCore<E>>();
		
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			double max = Double.POSITIVE_INFINITY;
			int maxNet = -1;
			List<SparseNeuralNetCore<E>> randomNets = allNets.get(netNumber);
			for (int numNet = 0; numNet < numNets; numNet++){
				if(randomNets.get(numNet).getValidationError() < max){
					max = randomNets.get(numNet).getValidationError();
					maxNet = numNet;
				}
			}
			
			randomNets.get(maxNet).save(this.getBaseDir() + "maxNNglobal" + netNumber);
			netsE.add(randomNets.get(maxNet));
		}
		
		nets = new ArrayList<NeuralNetCore<E>>();
		for(NeuralNetCore<E> nn : netsE)
			nets.add(nn);
	}

	protected SparseNeuralNetCore<E> createNet(String name, int seed) {
		return new SparseNeuralNetCore<E>(name, this.getBaseDir() + name
				+ "/", this, seed * 11, dropout, classify);
	}
	
	private Entry[] getTargets(DataSet<E> dataset) {
		Entry[] targets = new Entry[dataset.size()];
		if(classify){
			for(int i = 0; i < targets.length; i++)
				if(dataset.get(i).groundTruth != null) /* testing mode with no groundtruth. */
					targets[i] = new Entry(this.classes.getIndexFromFeature(dataset.get(i).groundTruth), 1.0);
		} else {
			for(int i = 0; i < targets.length; i++)	
				targets[i] = new Entry(0, dataset.get(i).outcome / this.maxOutcome);
		}
		return targets;
	}

	@Override
	public DataSet<E> test(){
		Entry[][] inputsTest = asEntryArrayMatrix(testDataSet, features);
		Entry[] targetsTest = getTargets(testDataSet);
		
		for(Instance each : this.testDataSet)
			for(int k = 0; k < this.getNumberOfOutputs(); k++)
				each.putResult("resultNode" + k, 0.0);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			SparseNeuralNetCore<E> net = netsE.get(netNumber);
			net.setTestData(inputsTest, targetsTest);
			net.test(this.testDataSet);
		}
		
		if(classify){
			for(Instance each : this.testDataSet){
				int maxIndex = -1;
				double maxValue = Double.NEGATIVE_INFINITY;
				for(int k = 0; k < this.getNumberOfOutputs(); k++){
					double value = each.getResult("resultNode" + k) / numNetsTotal;
					each.putResult("classProb" + classes.getFeatureByIndex(k), value);
					if(value > maxValue){
						maxValue = value;
						maxIndex = k;
					}
					each.removeResult("resultNode" + k);
				}
				each.label = classes.getFeatureByIndex(maxIndex);
				each.putResult("result", maxValue);
			}
		} else 
			for(Instance each : this.testDataSet)
				each.putResult("result", (each.getResult("resultNode0") / numNetsTotal) * this.maxOutcome);
		
		return this.testDataSet;
	}
	
	@Override
	public void load(String fileName) throws IOException{
		super.load(fileName);
		netsE = new ArrayList<SparseNeuralNetCore<E>>();
		for(int i = 0; i < (int)this.getDoubleParameter("numNetsTotal"); i++){
			SparseNeuralNetCore<E> net = createNet("net" + i, 11);
			net.load(fileName + i);
			netsE.add(net);
		}
	}

}
