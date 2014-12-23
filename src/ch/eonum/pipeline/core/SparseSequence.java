package ch.eonum.pipeline.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Sparse Sequence / Time series. Groundtruth can be either a class label, an
 * outcome or a sequence itself. The groundtruth sequence must have the same
 * length as the sequence. The data in the inherited data fields of Instance
 * represent the master data.
 * 
 * @author tim
 * 
 */
public class SparseSequence extends Sequence {
	/** time series data. */
	private List<Map<String, Double>> sequence;
	/**
	 * ground truth, if ground truth is a sequence. if not, the ground truth is
	 * stored in Instance.groundtruth.
	 */
	private List<List<Double>> groundTruthSequence;
	private List<List<Double>> sequenceResults;

	public SparseSequence(String id, String gt, Map<String, Double> vector) {
		super(id, gt, vector);
		this.sequence = new ArrayList<Map<String, Double>>();
	}
	
	/**
	 * copy constructor.
	 * @param seq
	 */
	public SparseSequence(SparseSequence seq) {
		super(seq.id, seq.groundTruth, new HashMap<String, Double>(seq.vector));
		this.className = seq.className;
		this.weight = seq.weight;
		this.outcome = seq.outcome;
		this.sequence = new ArrayList<Map<String, Double>>(seq.sequence);
		if(seq.groundTruthSequence != null)
			this.groundTruthSequence = new ArrayList<List<Double>>(seq.groundTruthSequence);
	}
	
	/**
	 * add a time point.
	 * @param point
	 */
	@Override
	public void addTimePoint(Map<String, Double> point){
		this.sequence.add(point);
	}
	
	/**
	 * Get the sequence length / number of time steps.
	 * @return
	 */
	@Override
	public int getSequenceLength(){
		return this.sequence.size();
	}
	
	/**
	 * Set the value of a feature of a time point.
	 * @param t time point
	 * @param feature
	 * @param value
	 */
	@Override
	public void put(int t, String feature, double value){
		this.sequence.get(t).put(feature, value);
	}
	
	/**
	 * get a feature from time point t:
	 * @param t
	 * @param feature
	 * @return
	 */
	@Override
	public double get(int t, String feature){
		return this.sequence.get(t).containsKey(feature) ? this.sequence.get(t)
				.get(feature) : 0.0;
	}

	/**
	 * 
	 * @param t time
	 * @param f output feature
	 * @return
	 */
	@Override
	public double groundTruthAt(int t, int f) {
		return groundTruthSequence.get(t).get(f);
	}
	
	/**
	 * initialize the groundtruth sequence.
	 */
	@Override
	public void initGroundTruthSequence() {
		this.groundTruthSequence = new ArrayList<List<Double>>();
	}

	/**
	 * set the ground truth at point (t|f)
	 * @param t time point
	 * @param f output feature
	 * @param result value
	 */
	@Override
	public void addGroundTruth(int t, int f, double result) {
		this.groundTruthSequence.get(t).set(f, result);
	}
	
	/**
	 * add a time point in the ground truth sequence.
	 * 
	 * @param groundTruth
	 */
	public void addGroundTruth(List<Double> groundTruth) {
		this.groundTruthSequence.add(groundTruth);
	}

	/**
	 * Sort the time points according the specified feature.
	 * @param feature
	 */
	@Override
	public void sortTimePoints(final String feature) {
		Collections.sort(this.sequence, new Comparator<Map<String, Double>>(){
			@Override
			public int compare(Map<String, Double> o1, Map<String, Double> o2) {
				return o1.get(feature).compareTo(o2.get(feature));
			}
			
		});
	}
	
	/**
	 * add a copy of the sequence at the end of the sequence. Some extra context
	 * in both directions can be obtained by this.
	 */
	@Override
	public void doubleSequence(){
		int l = sequence.size();
		for (int i = 0; i < l; i++)
			sequence.add(sequence.get(i));
	}
	
	@Override
	public Set<String> features(){
		Set<String> f = new LinkedHashSet<String>();
		for(Map<String, Double> each : this.sequence)
			f.addAll(each.keySet());
		return f;
	}

	@Override
	public void levelAverage() {
		Set<String> sequenceFeatures = new HashSet<String>();
		for(Map<String, Double> each : this.sequence)
			for(String feature : each.keySet()){
				String f = "m" + feature;
				this.put(f, this.get(f) + each.get(feature));
				sequenceFeatures.add(f);
			}
		for(String feature : sequenceFeatures)
			this.put(feature, this.get(feature)/this.sequence.size());
		this.put("seqLength", this.sequence.size());
	}

	@Override
	public void levelMax() {
		for(Map<String, Double> each : this.sequence)
			for(String feature : each.keySet()){
				this.put("max" + feature,
						Math.max(this.get("max" + feature), each.get(feature)));
			}
	}

	@Override
	public void levelMin() {
		for (Map<String, Double> each : this.sequence)
			for (String feature : each.keySet()) {
				if (this.get("min" + feature) == 0.0)
					this.put("min" + feature, each.get(feature));
				else
					this.put(
							"min" + feature,
							Math.min(
									0.1,
									Math.min(this.get("min" + feature),
											each.get(feature))));
			}
		this.put("seqLength", this.sequence.size());
	}
	
	/**
	 * standard deviation
	 */
	@Override
	public void levelStd() {
		Set<String> sequenceFeatures = new HashSet<String>();
		Map<String, Double> avgs = new HashMap<String, Double>();
		for(Map<String, Double> each : this.sequence)
			for(String feature : each.keySet()){
				if(!avgs.containsKey(feature))
					avgs.put(feature, 0.0);
				avgs.put(feature, avgs.get(feature) + each.get(feature));
				sequenceFeatures.add(feature);
			}
		for(String feature : sequenceFeatures)
			avgs.put(feature, avgs.get(feature)/this.sequence.size());
		sequenceFeatures = new HashSet<String>();
		for(Map<String, Double> each : this.sequence)
			for(String feature : each.keySet()){
				String f = "std" + feature;
				this.put(f, this.get(f) + Math.pow(each.get(feature)-avgs.get(feature),2.0));
				sequenceFeatures.add(f);
			}
		for(String feature : sequenceFeatures)
			this.put(feature, Math.sqrt(this.get(feature)/this.sequence.size()));
	}
	
	/**
	 * get the normed (by sequence length) center of mass of all features in the given features object.
	 * @param features
	 */
	@Override
	public void levelCenterOfMass(Features features) {
			for (int f = 0; f < features.size(); f++) {
				String feature = features.getFeatureByIndex(f);
				double totalMass = 0.0;
				double com = 0.0;
				int position = 0;
				for (Map<String, Double> each : this.sequence){
					double value = each.containsKey(feature) ? each.get(feature) : 0.0;
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
	
	/**
	 * create a new feature for each point in time and feature. This is only
	 * feasible for fixed length and small sequences. (e.g. digit images 32x32
	 * pixels)
	 */
	@Override
	public void levelTimeWindow() {
		for(int i = 0; i < sequence.size(); i++){
			for(String feature : sequence.get(i).keySet())
				put(i + feature, sequence.get(i).get(feature));
		}
	}

	@Override
	public void deleteSequence() {
		this.sequence = new ArrayList<Map<String, Double>>();
	}

	/**
	 * Push all information in the flat instance to all data points
	 * in the time series.
	 */
	@Override
	public void pushUp() {
		for(String feature : vector.keySet())
			for(Map<String, Double> point : this.sequence)
				point.put(feature, this.get(feature));
	}

	/**
	 * Get a data set where each time point is a single instance. All data is
	 * copied and no references to this sequence being kept.
	 * 
	 * @return
	 */
	@Override
	public DataSet<SparseInstance> createDataSetFromTimePoints() {
		DataSet<SparseInstance> set = new DataSet<SparseInstance>();
		for(Map<String, Double> point : this.sequence){
			HashMap<String, Double> pointCopy = new HashMap<String, Double>();
			for(String each : point.keySet())
				pointCopy.put(each, point.get(each));
			SparseInstance inst = new SparseInstance(this.id, this.groundTruth, pointCopy, this.className);
			inst.outcome = outcome;
			set.addInstance(inst);
		}
		return set;
	}
	
	/**
	 * Get a data set where each time point is a single instance. The instances
	 * hold the same feature maps as the time points. No copies are beigin made.
	 * 
	 * You can use this to apply basic feature transformers. (Normalization, PCA
	 * ..) or to speed up a write-only process.
	 * 
	 * @return
	 */
	@Override
	public DataSet<SparseInstance> getDataSetFromTimePoints() {
		DataSet<SparseInstance> set = new DataSet<SparseInstance>();
		for(Map<String, Double> point : this.sequence){
			SparseInstance inst = new SparseInstance(this.id, this.groundTruth, point, this.className);
			inst.outcome = outcome;
			set.addInstance(inst);
		}
		return set;
	}

	@Override
	public void invertTemporalOrder() {
		List<Map<String, Double>> newSequence = new ArrayList<Map<String, Double>>();
		for(int i = this.sequence.size() - 1; i >= 0; i--)
			newSequence.add(sequence.get(i));
		this.sequence = newSequence;
	}

	@Override
	public void pushMasterOnTop() {
		this.sequence.add(this.vector);
	}

	@Override
	public void putMasterAtBottom() {
		this.sequence.add(0, this.vector);
	}

	@Override
	public void shuffle() {
		Collections.shuffle(this.sequence);
	}

	/**
	 * group / summarize all time points with the same value
	 * of feature groupFeature.
	 * After grouping, the sequence is sorted according groupFeature.
	 * 
	 * @param groupFeature
	 */
	@Override
	public void groupBy(String groupFeature) {
		List<Map<String, Double>> seq = this.sequence;
		this.sequence = new ArrayList<Map<String, Double>>();
		for(Map<String, Double> pointAdd : seq){
			boolean found = false;
			for(Map<String, Double> point : this.sequence){
				if(Math.abs(pointAdd.get(groupFeature) - point.get(groupFeature)) < 0.00000001){
					found = true;
					for(String feature : pointAdd.keySet()){
						if (!feature.equals(groupFeature))
							if (point.containsKey(feature))
								point.put(feature, pointAdd.get(feature)
										+ point.get(feature));
							else
								point.put(feature, pointAdd.get(feature));
					}
				}
			}
			if(!found)
				this.sequence.add(pointAdd);
		}
		this.sortTimePoints(groupFeature);
	}

	@Override
	public void clearSequence() {
		this.sequence = new ArrayList<Map<String, Double>>();
	}
	
	@Override
	public String toString(){
		String seq = "";
		if (this.groundTruthSequence != null
				&& this.groundTruthSequence.size() != this.sequence.size())
			return "Invalid format: sequence size is " + this.sequence.size()
					+ " and groundtruth sequence size is "
					+ this.groundTruthSequence.size();
		for(int i = 0; i < this.sequence.size(); i++){
			Map<String, Double> point = this.sequence.get(i);
			seq += point;
			if(this.groundTruthSequence != null)
				seq += " => " + this.groundTruthSequence.get(i);
			seq += "\n";
		}
		return "master: " + super.toString() + "\n" + seq;
	}
	
	@Override
	public SparseSequence copy() {
		Map<String, Double> newVector = new HashMap<String, Double>(vector);
		SparseSequence inst = new SparseSequence(id, groundTruth, newVector);
		inst.sequence = new ArrayList<Map<String, Double>>();
		for(Map<String, Double> point : sequence){
			inst.sequence.add(new HashMap<String, Double>(point));
		}
		inst.groundTruthSequence = new ArrayList<List<Double>>();
		for(List<Double> e : groundTruthSequence)
			inst.groundTruthSequence.add(new ArrayList<Double>(e));
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
		double value;
		List<Map<String, Double>> instSeq = ((SparseSequence)instance).sequence;
		Random rand = new Random(1234);
		List<Map<String, Double>> copy = new ArrayList<Map<String, Double>>(sequence);
		for(Map<String, Double> point : sequence){
			Map<String, Double> point2 = instSeq.get(rand.nextInt(instSeq.size()));
			value = point2.containsKey(feature) ? point2.get(feature) : 0;
			point.put(feature, value);
		}
		for(Map<String, Double> point : instSeq){
			Map<String, Double> point2 = copy.get(rand.nextInt(copy.size()));
			value = point2.containsKey(feature) ? point2.get(feature) : 0;
			point.put(feature, value);
		}
		
	}
	
	@Override
	public double remove(String feature) {
		for(Map<String, Double> point : this.sequence)
			point.remove(feature);
		return super.remove(feature);
	}

	/**
	 * initialize the sequence results in the proper size and filled with NaN.
	 */
	@Override
	public void initSequenceResults() {
		this.sequenceResults = new ArrayList<List<Double>>();
		for(int i = 0; i < this.groundTruthSequence.size(); i++){
			List<Double> l = new ArrayList<Double>();
			for(int j = 0; j < this.groundTruthSequence.get(i).size(); j++)
				l.add(Double.NaN);
			this.sequenceResults.add(l);
		}
	}

	/**
	 * set the result at point (t|f)
	 * @param t time point
	 * @param f output feature
	 * @param result value
	 */
	@Override
	public void addSequenceResult(int t, int f, double result) {
		if(sequenceResults != null)
			this.sequenceResults.get(t).set(f, result);
	}

	/**
	 * get the sequence result at time point t and output feature f
	 * 
	 * @param t
	 * @param f
	 * @return
	 */
	@Override
	public double resultAt(int t, int f) {
		return this.sequenceResults.get(t).get(f);
	}

	/**
	 * get a dense representation of the input sequence.
	 * @param features
	 * @return
	 */
	@Override
	public double[][] getDenseRepresentation(Features features) {
		int length = Math.max(1, this.getSequenceLength());
		double[][] data = new double[length][features.size()];
		for(int t = 0; t < this.getSequenceLength(); t++){
			Map<String, Double> pointT = this.sequence.get(t);
			for(String feature : pointT.keySet())
				if(features.hasFeature(feature))
					data[t][features.getIndexFromFeature(feature)] = pointT.get(feature);
		}
		return data;
	}

	@Override
	public boolean hasSequenceTarget() {
		return (groundTruthSequence != null && groundTruthSequence.size() == sequence.size());
	}
	
	/**
	 * create a target sequence for each sequence by copying the sequence
	 * with a given time lag. Hence we can can train to predict the sequence.
	 * The first timeLag points will be NaN, because we cannot predict anything
	 * yet.
	 * 
	 * @param timeLag
	 *            in number of points in time.
	 * @param features 
	 */
	@Override
	public void createTargetForPrediction(int timeLag, Features features) {
		this.groundTruthSequence = new ArrayList<List<Double>>();
		int length = features.size();
		if(timeLag > this.sequence.size())
			timeLag = this.sequence.size();	
		for(int i =  timeLag; i < sequence.size(); i++){
			List<Double> p = new ArrayList<Double>();
			for(int j = 0; j < length; j++)
				p.add(sequence.get(i).get(features.getFeatureByIndex(j)));
			this.groundTruthSequence.add(p);
		}
		for(int i = 0; i < timeLag; i++){
			List<Double> p = new ArrayList<Double>();
			for(int j = 0; j < length; j++)
				p.add(Double.NaN);
			this.groundTruthSequence.add(p);
		}
		assert(this.groundTruthSequence.size() == this.sequence.size());
	}

	/**
	 * number of outputs. This is the number of classes in a classification
	 * task.
	 * 
	 * @return
	 */
	@Override
	public int outputSize() {
		if(groundTruthSequence == null)
			return 1;
		return this.groundTruthSequence.get(0).size();
	}

	/**
	 * Do we have a ground truth sequence. 
	 * @return
	 */
	@Override
	public boolean hasGroundTruthSequence() {
		return this.groundTruthSequence != null;
	}

	@Override
	public int getGroundTruthLength() {
		if(this.groundTruthSequence == null)
			return -1;
		return this.groundTruthSequence.size();
	}

	/**
	 * remove time point t in input and groundtruth.
	 * @param i
	 */
	public void removeSequenceElementAt(int t) {
		this.groundTruthSequence.remove(t);
		this.sequence.remove(t);
	}
	
	/**
	 * Get a sparse representation of the input features using Entries.
	 * @param features
	 * @param bias add an addditional feature with value 1.0
	 * @return
	 */
	public Entry[][] getSparseRepresentation(Features features, boolean bias){
		Entry[][] entries = new Entry[this.sequence.size()][];
		int i = 0;
		for(Map<String, Double> e : this.sequence){
			Set<String> set = new HashSet<String>();
			for (String feature : e.keySet())
				if(features.hasFeature(feature))
					set.add(feature);
			Entry[] values = new Entry[set.size() + (bias ? 1 : 0)];
			int f = 0;
			for (String feature : set)
				values[f++] = new Entry(features.getIndexFromFeature(feature), e.get(feature));
			if(bias)
				values[values.length - 1] = new Entry(features.size(), 1.0);
			entries[i++] = values;
		}
		return entries;		
	}

	public Map<String, Double> getTimePoint(int i) {
		return this.sequence.get(i);
	}

}
