package ch.eonum.pipeline.core;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Dense Sequence / Time series. The ground truth can be either a class label,
 * an outcome or a sequence itself. The ground truth sequence must have the same
 * length as the sequence. The data in the inherited data fields of Instance
 * represent the master data. The master data is always sparse. Some methods
 * which can be used in the sparse sequence, such as adding elements to the
 * sequence, are disabled in order to provide better performance.
 * 
 * See also {@link Sequence}
 * 
 * @author tim
 * 
 */
public class DenseSequence extends Sequence {
	/** time series data. */
	private double[][] sequence;
	/**
	 * ground truth, if ground truth is a sequence. if not, the ground truth is
	 * stored in Instance.groundtruth.
	 */
	private double[][] groundTruthSequence;
	private double[][] sequenceResults;
	private Features features;

	/**
	 * constructor with master data.
	 * 
	 * @param id
	 * @param groundTruth
	 * @param masterData
	 * @param features
	 */
	public DenseSequence(String id, String groundTruth,
			Map<String, Double> masterData, Features features) {
		super(id, groundTruth, masterData);
		this.features = features;
	}

	/**
	 * constructor without master data.
	 * 
	 * @param id
	 * @param groundTruth
	 * @param features
	 */
	public DenseSequence(String id, String groundTruth, Features features) {
		this(id, groundTruth, new HashMap<String, Double>(), features);
	}

	/**
	 * copy constructor. !! only shallow copy !!
	 * 
	 * @param seq
	 */
	public DenseSequence(DenseSequence seq) {
		super(seq.id, seq.groundTruth, new HashMap<String, Double>(seq.vector));
		this.className = seq.className;
		this.weight = seq.weight;
		this.outcome = seq.outcome;
		this.sequence = seq.sequence;
		if(seq.groundTruthSequence != null)
			this.groundTruthSequence = seq.groundTruthSequence;
	}
	
	/**
	 * Add a new time point. This method is not very efficient using dense
	 * sequences.. Please use initSequence and then put(t, f, v) if you know the
	 * sequence length.
	 * 
	 * @param point
	 */
	@Override
	public void addTimePoint(Map<String, Double> point){
		double[] pointArray = new double[features.size()];
		for(int i = 0; i < features.size(); i++)
			if(point.containsKey(features.getFeatureByIndex(i)))
				pointArray[i] = point.get(features.getFeatureByIndex(i));
		this.addTimePoint(pointArray);
	}
	
	/**
	 * Add a new time point. This method is not very efficient using dense
	 * sequences. Please use initSequence and then put(t, f, v) if you know the
	 * sequence length.
	 * 
	 * @param point
	 */
	public void addTimePoint(double[] point) {
		if(this.sequence == null)
			sequence = new double[0][];
		double[][] newSequence = new double[sequence.length + 1][];
		for(int i = 0; i < this.sequence.length; i++)
			newSequence[i] = sequence[i];
		newSequence[sequence.length] = point;
		this.sequence = newSequence;
	}
	
	/**
	 * initialize a sequence with length size and all values set to NaN
	 * 
	 * @param size
	 */
	public void initSequence(int size) {
		this.sequence = new double[size][features.size()];
		for(int i = 0; i < sequence.length; i++)
			for(int j = 0; j < sequence[i].length; j++)
				sequence[i][j] = Double.NaN;
		
	}
	
	/**
	 * Get the sequence length / number of time steps.
	 * @return
	 */
	@Override
	public int getSequenceLength(){
		return this.sequence.length;
	}
	
	/**
	 * Set the value of a feature of a time point.
	 * @param t time point
	 * @param feature name
	 * @param value
	 */
	@Override
	public void put(int t, String feature, double value){
		this.sequence[t][features.getIndexFromFeature(feature)] = value;
	}
	
	/**
	 * Set the value of a feature of a time point.
	 * @param t time point
	 * @param feature index
	 * @param value
	 */
	public void put(int t, int feature, double value){
		this.sequence[t][feature] = value;
	}
	
	@Override
	public double get(int t, String feature){
		return this.sequence[t][features.getIndexFromFeature(feature)];
	}

	@Override
	public double groundTruthAt(int t, int f) {
		return groundTruthSequence[t][f];
	}

	@Override
	public int outputSize() {
		if(groundTruthSequence == null)
			return 1;
		return this.groundTruthSequence[0].length;
	}
	
	@Override
	public boolean hasGroundTruthSequence() {
		return this.groundTruthSequence != null;
	}
	
	@Override
	public int getGroundTruthLength() {
		if(this.groundTruthSequence == null)
			return -1;
		return this.groundTruthSequence.length;
	}
	
	@Override
	public Set<String> features(){
		Set<String> f = new LinkedHashSet<String>();
		for(int i = 0; i < features.size(); i++)
			f.add(features.getFeatureByIndex(i));
		return f;
	}

	/**
	 * Sort the time points according the specified feature in ascending order.
	 * @param feature
	 */
	@Override
	public void sortTimePoints(final String feature) {
		List<double[]> c = Arrays.asList(sequence);
		final int i = features.getIndexFromFeature(feature);
		Collections.sort(c, new Comparator<double[]>(){
			@Override
			public int compare(double[] o1, double[] o2) {
				return ((Double)o1[i]).compareTo(o2[i]);
			}
		});
		for(int h = 0; h < c.size(); h++)
			sequence[h] = c.get(h);
	}
	
	@Override
	public void doubleSequence(){
		double[][] newSeq = new double[sequence.length*2][sequence[0].length];
		for (int i = 0; i < sequence.length; i++){
			newSeq[i] = sequence[i];
			newSeq[i*2] = sequence[i];
		}
		sequence = newSeq;
	}

	@Override
	public void levelAverage() {
		for(double[] each : this.sequence)
			for(int i = 0; i < each.length; i++){
				String f = "m" + features.getFeatureByIndex(i);
				this.put(f, this.get(f) + each[i]);
			}
		for (int f = 0; f < features.size(); f++)
			this.put("m" + features.getFeatureByIndex(f),
					this.get("m" + features.getFeatureByIndex(f))
							/ this.sequence.length);
		this.put("seqLength", this.sequence.length);
	}

	@Override
	public void levelMax() {
		for(double[] each : this.sequence)
			for(int i = 0; i < each.length; i++){
				String f = "max" + features.getFeatureByIndex(i);
				this.put(f, Math.max(this.get(f), each[i]));
			}
	}

	@Override
	public void levelMin() {
		for(double[] each : this.sequence)
			for(int i = 0; i < each.length; i++){
				String f = "min" + features.getFeatureByIndex(i);
				this.put(f, Math.min(this.get(f), each[i]));
			}
	}
	
	/**
	 * standard deviation
	 */
	@Override
	public void levelStd() {
		Map<String, Double> avgs = new HashMap<String, Double>();
		for(double[] each : this.sequence)
			for(int i = 0; i < each.length; i++){
				String f = features.getFeatureByIndex(i);
				if(!avgs.containsKey(f))
					avgs.put(f, 0.0);
				avgs.put(f, avgs.get(f) + each[i]);
			}
		for(int f = 0; f < features.size(); f++){
			String feature = features.getFeatureByIndex(f);
			avgs.put(feature, avgs.get(feature)/this.sequence.length);
		}
			
		for(double[] each : this.sequence)
			for(int f = 0; f < features.size(); f++){
				String feature = "std" + features.getFeatureByIndex(f);
				this.put(feature, this.get(feature) + Math.pow(each[f] - avgs.get(feature), 2.0));
			}
		for(int f = 0; f < features.size(); f++){
			String feature = features.getFeatureByIndex(f);
			this.put(feature, Math.sqrt(this.get(feature)/this.sequence.length));
		}
	}
	
	/**
	 * get the normed (by sequence length) center of mass of all features in the
	 * given features object.
	 * 
	 * @param features
	 */
	@Override
	public void levelCenterOfMass(Features features) {
			for (int f = 0; f < features.size(); f++) {
				String feature = features.getFeatureByIndex(f);
				double totalMass = 0.0;
				double com = 0.0;
				int position = 0;
				for(double[] each : this.sequence){
					double value = each[f];
					totalMass += value;
					com += value * position;
					position++;
				}
				if(totalMass == 0.0)
					com = 0.;
				else {
					com /= totalMass;
					com /= position;
				}
				put("com" + feature, com);
			}
	}
	
	@Override
	public void levelTimeWindow() {
		for(int i = 0; i < sequence.length; i++){
			for (int f = 0; f < features.size(); f++) {
				String feature = features.getFeatureByIndex(f);
				put(i + feature, sequence[i][f]);
			}
		}
	}
	
	/**
	 * #TODO reverse the order of the time lag
	 * #TODO This function does not work yet, if the feature set for the targets
	 * is different than the feature set of the inputs.
	 */
	@Override
	public void createTargetForPrediction(int timeLag, Features features) {
		int length = features.size();
		this.groundTruthSequence = new double[sequence.length][length];
		if(timeLag > this.sequence.length)
			timeLag = this.sequence.length;
		for(int i = 0; i < timeLag; i++)
			for(int j = 0; j < length; j++)
				this.groundTruthSequence[i][j] = Double.NaN;
		for(int i = timeLag; i < sequence.length; i++)
			for(int j = 0; j < length; j++)
				this.groundTruthSequence[i][j] = sequence[i - timeLag][j];
			
		assert(this.groundTruthSequence.length == this.sequence.length);
	}

	/**
	 * Get a data set where each time point is a single instance. All data is
	 * copied and no references to this sequence is kept.
	 * 
	 * @return
	 */
	@Override
	public DataSet<SparseInstance> createDataSetFromTimePoints() {
		DataSet<SparseInstance> set = new DataSet<SparseInstance>();
		for(int i = 0; i < sequence.length; i++){
			HashMap<String, Double> pointCopy = new HashMap<String, Double>();
			for (int f = 0; f < features.size(); f++) {
				String feature = features.getFeatureByIndex(f);
				pointCopy.put(feature, sequence[i][f]);
			}
			SparseInstance inst = new SparseInstance(this.id, this.groundTruth, pointCopy, this.className);
			inst.outcome = outcome;
			set.addInstance(inst);
		}
		return set;
	}
	
	/**
	 * Get a data set where each time point is a single instance. The instances
	 * hold the same feature maps as the time points.
	 * 
	 * You can use this to apply basic feature transformers. (Normalization, PCA
	 * ..)
	 * 
	 * @return
	 */
	@Override
	public DataSet<SparseInstance> getDataSetFromTimePoints() {
		throw new UnsupportedOperationException("DenseSequence does not support this operation");
	}

	@Override
	public void invertTemporalOrder() {
		double[][] newSequence = new double[sequence.length][];
		int j = 0;
		for(int i = this.sequence.length - 1; i >= 0; i--)
			newSequence[j++] = sequence[i];
		this.sequence = newSequence;
	}

	@Override
	public void pushMasterOnTop() {
		double[][] newSequence = new double[sequence.length + 1][];
		for(int i = 0; i < this.sequence.length; i++)
			newSequence[i] = sequence[i];
		for (int f = 0; f < features.size(); f++)
			newSequence[sequence.length][f] = this.get(features.getFeatureByIndex(f));
		this.sequence = newSequence;
	}

	@Override
	public void putMasterAtBottom() {
		double[][] newSequence = new double[sequence.length + 1][];
		for(int i = 1; i < this.sequence.length + 1; i++)
			newSequence[i] = sequence[i];
		for (int f = 0; f < features.size(); f++)
			newSequence[0][f] = this.get(features.getFeatureByIndex(f));
		this.sequence = newSequence;
	}

	@Override
	public void shuffle() {
		Collections.shuffle(Arrays.asList(this.sequence));
	}

	/**
	 * group / add all time points with the same value of feature groupFeature.
	 * After grouping, the sequence is sorted according groupFeature.
	 * 
	 * @param groupFeature
	 */
	@Override
	public void groupBy(String groupFeature) {
		int f = features.getIndexFromFeature(groupFeature);
		double[][] seq = this.sequence;
		double[][] tempSequence = new double[seq.length][features.size()];
		int iTemp = 0;
		for(double[] pointAdd : seq){
			boolean found = false;
			for(int tp = 0; tp < iTemp; tp++){
				if(Math.abs(pointAdd[f] - tempSequence[tp][f]) < 0.00000001){
					found = true;
					for(int i = 0; i < pointAdd.length; i++) {
						if (f != i)
							tempSequence[tp][i] += pointAdd[i];
					}
					break;
				}
			}
			if(!found)
				tempSequence[iTemp++] = pointAdd;
		}
		
		sequence = new double[iTemp][features.size()];
		for(int i = 0; i < iTemp; i++)
			sequence[i] = tempSequence[i];
		
		this.sortTimePoints(groupFeature);
	}

	@Override
	public void clearSequence() {
		this.sequence = null;
	}
	
	@Override
	public String toString(){
		String seq = "";
		if (this.groundTruthSequence != null
				&& this.groundTruthSequence.length != this.sequence.length)
			return "Invalid format: sequence size is " + this.sequence.length
					+ " and groundtruth sequence size is "
					+ this.groundTruthSequence.length;
		for(int i = 0; i < this.sequence.length; i++){
			seq += Arrays.toString(this.sequence[i]);
			if(this.groundTruthSequence != null)
				seq += " => " + Arrays.toString(this.groundTruthSequence[i]);
			seq += "\n";
		}
		return "master: " + super.toString() + "\n" + seq;
	}
	
	@Override
	public DenseSequence copy() {
		Map<String, Double> newVector = new HashMap<String, Double>(vector);
		DenseSequence inst = new DenseSequence(id, groundTruth, newVector,
				features);
		inst.sequence = new double[sequence.length][features.size()];
		for (int p = 0; p < sequence.length; p++)
			inst.sequence[p] = Arrays.copyOf(sequence[p], sequence[p].length);
		if (groundTruthSequence != null) {
			inst.groundTruthSequence = new double[groundTruthSequence.length][groundTruthSequence[0].length];
			for (int p = 0; p < groundTruthSequence.length; p++)
				inst.groundTruthSequence[p] = Arrays.copyOf(
						groundTruthSequence[p], groundTruthSequence[p].length);
		}
		inst.outcome = outcome;
		return inst;
	}
	
	/**
	 * exchange a certain feature with another instance.
	 * Used for randomization (random permutation) of certain features.
	 * @param feature
	 * @param instance
	 */
	@Override
	public void exchangeFeature(String feature, Instance instance) {
		super.exchangeFeature(feature, instance);
		int f = features.getIndexFromFeature(feature);
		double value;
		double[][] instSeq = ((DenseSequence)instance).sequence;
		Random rand = new Random(1234);
		double[][] copy = new double[sequence.length][features.size()];
		for (int p = 0; p < sequence.length; p++)
			copy[p] = Arrays.copyOf(sequence[p], sequence[p].length);
		
		for(int i = 0; i < sequence.length; i++) {
			double[] point2 = instSeq[rand.nextInt(instSeq.length)];
			value = point2[f];
			sequence[i][f] = value;
		}
		for(int i = 0; i < instSeq.length; i++) {
			double[] point2 = sequence[rand.nextInt(sequence.length)];
			value = point2[f];
			instSeq[i][f] = value;
		}
	}
	
	@Override
	public double remove(String feature) {
		for(double[] point : this.sequence)
			if(features.hasFeature(feature))
				point[features.getIndexFromFeature(feature)] = 0;
		return super.remove(feature);
	}

	@Override
	public void initSequenceResults() {
		this.sequenceResults = new double[this.groundTruthSequence.length][this.outputSize()];
		for(int i = 0; i < sequenceResults.length; i++)
			for(int j = 0; j < sequenceResults[i].length; j++)
				sequenceResults[i][j] = Double.NaN;
	}

	@Override
	public void addSequenceResult(int t, int f, double result) {
		if(this.sequenceResults != null)
			this.sequenceResults[t][f] = result;
	}
	
	@Override
	public double resultAt(int t, int f) {
		return this.sequenceResults[t][f];
	}

	@Override
	public double[][] getDenseRepresentation(Features features) {
		if(features.size() != this.features.size())
			throw new IllegalArgumentException("Error: feature size does not match");
		return sequence;
	}

	@Override
	public boolean hasSequenceTarget() {
		return (groundTruthSequence != null && groundTruthSequence.length == sequence.length);
	}

	@Override
	public void pushUp() {
		for(String feature : vector.keySet())
			for(double[] point : this.sequence)
				point[features.getIndexFromFeature(feature)] = this.get(feature);
	}

	@Override
	public void deleteSequence() {
		this.sequence = null;
	}

	@Override
	public void addGroundTruth(int t, int f, double result) {
		this.groundTruthSequence[t][f] = result;
	}

	@Override
	public void initGroundTruthSequence() {
		this.groundTruthSequence = new double[this.sequence.length][this.outputSize()];
		for(int i = 0; i < groundTruthSequence.length; i++)
			for(int j = 0; j < groundTruthSequence[i].length; j++)
				groundTruthSequence[i][j] = Double.NaN;
	}

}
