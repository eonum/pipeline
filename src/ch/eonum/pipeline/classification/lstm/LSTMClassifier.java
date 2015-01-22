package ch.eonum.pipeline.classification.lstm;


import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Sequence;

/**
 * Long Short Term Memory Recurrent Neural Network for classification.
 * @author tim
 *
 * @param <E>
 */
public class LSTMClassifier<E extends Sequence> extends LSTM<E> {
	private boolean targetSequence = false;
	
	public LSTMClassifier(){
		super();
		classify = true;
	}
	
	@Override
	public void train(){
		if(!targetSequence )
			this.prepareClasses();
		super.train();
	}
	
	/**
	 * set the classes. use this, if your class labels are not in the master
	 * data, but in the target sequence.
	 * 
	 * @param features
	 */
	public void setClasses(Features features){
		this.targetSequence = true;
		this.classes = features;
	}
	
	@Override
	protected double[][][] loadTargets(DataSet<E> set) {
		double data[][][] = new double[set.size()][][];
		for(int i = 0; i < set.size(); i ++){
			Sequence seq = set.get(i);
			int length = Math.max(1, seq.getSequenceLength());
			
			data[i] = new double[length][classes.size()];
	
			if(seq.hasSequenceTarget()){
				if (!this.targetSequence)
					throw new AssertionError(
							"Sequence has target sequence, but no target sequence classes are specified.");
				if (classes.size() != seq.outputSize())
					throw new AssertionError(
							"Number of classes does not fit the number of outputs for sequence: "
									+ seq);
				for(int j = 0; j < length; j++)
					for(int k = 0; k < classes.size(); k++)
						data[i][j][k] = seq.groundTruthAt(j, k);
			} else {
				for(int j = 0; j < length; j++)
					for(int k = 0; k < data[i][j].length; k++)
						data[i][j][k] = Double.NaN;
				for (int k = 0; k < data[i][length - 1].length; k++)
					data[i][length - 1][k] = 0.0;
				data[i][length-1][classes.getIndexFromFeature(seq.groundTruth)] = 1.0;
			}
		}
		return data;
	}
	
	@Override
	protected void fillResults(int numNetsTotal) {
		for(Instance each : this.testDataSet){
			double maxProb = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < classes.size(); i++){
				if(each.getResult("classProb" + i) > maxProb){
					each.label = classes.getFeatureByIndex(i);
					maxProb = each.getResult("classProb" + i);
				}
				each.putResult("classProb" + i, each.getResult("classProb" + i) / numNetsTotal);
			}
			each.putResult("result", maxProb / numNetsTotal);
		}
	}
	
	/**
	 * set the result of one net for a sequence
	 * @param seq
	 * @param outputUnit
	 * @param activation
	 */
	@Override
	public synchronized void setResults(Sequence seq, int outputUnit,
			double activation, String threadName) {
		seq.putResult("classProb" + outputUnit, seq.getResult("classProb" + outputUnit)
				+ activation);
	}
	
	@Override
	protected LSTMCore<E> createNet(String name, String folder, int seed) {
		LSTMCore<E> core = super.createNet(name, folder, seed);
		core.useSoftMaxOutputLayer();
		return core;
	}

	
}
