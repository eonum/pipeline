package ch.eonum.pipeline.core;

import java.util.Map;
import java.util.Set;

/**
 * A sequence can represent a time series, a sequence of events, a string of
 * connected data.. The ground truth of a sequence can be a sequence as well
 * (with the same length) or can be a single data point with one or more
 * dimensions or can be a class label or a sequence of class labels.
 * 
 * A sequence is an instance and can therefore contain additional information
 * about the sequence as a whole.
 * 
 * A sequence can be used with sequence classifiers such as
 * {@link ch.eonum.pipeline.classification.GaussianMixtureModel} or
 * {@link ch.eonum.pipeline.classification.lstm.LSTM} or the sequence can be
 * leveled (summarizing information in the sequence to one single data point)
 * using for instance the methods in {@link SequenceDataSet} and then be applied
 * to any standard instance classifier.
 * 
 * @author tim
 * 
 */
public abstract class Sequence extends SparseInstance {
	
	public Sequence(String id, String gt, Map<String, Double> vector) {
		super(id, gt, vector);
	}

	public Sequence(String id, String groundTruth, Map<String, Double> vector,
			String className) {
		super(id, groundTruth, vector, className);
	}

	/**
	 * Get the length of the ground truth sequence. Returns -1 if there is no
	 * ground truth sequence.
	 * 
	 * @return
	 */
	public abstract int getGroundTruthLength();

	/**
	 * Return true if there is a ground truth sequence.
	 * @return
	 */
	public abstract boolean hasGroundTruthSequence();

	/**
	 * Get the number of dimensions in the output/ground truth sequence. Returns
	 * 1 if there is no ground truth sequence.
	 * 
	 * @return
	 */
	public abstract int outputSize();

	/**
	 * Create a ground truth sequence for standard prediction tasks where the
	 * outcome should be identical to the future input. E.g. Weather forecast
	 * based on current weather.
	 * 
	 * @param timeLag number of time points we want to look into the future.
	 * @param features features that should be predicted. (>= 1)
	 */
	public abstract void createTargetForPrediction(int timeLag, Features features);

	/**
	 * Return true if there is a ground truth sequence and the ground truth
	 * sequence has the same length as the input sequence.
	 * 
	 * @return
	 */
	public abstract boolean hasSequenceTarget();

	/**
	 * Get a dense double matrix representation of the sequence data.
	 * @param features
	 * @return
	 */
	public abstract double[][] getDenseRepresentation(Features features);

	/**
	 * Get the value for feature f at time point t.
	 * @param t
	 * @param f
	 * @return
	 */
	public abstract double resultAt(int t, int f);

	/**
	 * Set the value of feature f at time point t. The sequence length must be
	 * at least t+1. The existing value will be overwritten.
	 * 
	 * @param t
	 * @param f
	 * @param value
	 */
	public abstract void addSequenceResult(int t, int f,
			double value);

	/**
	 * Initialize a sequence with results with the same size as the ground truth
	 * sequence. All values are NaN.
	 */
	public abstract void initSequenceResults();

	/**
	 * Deep copy.
	 */
	public abstract Sequence copy();

	public abstract void clearSequence();

	public abstract void groupBy(String groupFeature);

	public abstract void shuffle();

	public abstract void putMasterAtBottom();

	public abstract void pushMasterOnTop();

	public abstract void invertTemporalOrder();

	public abstract DataSet<SparseInstance> getDataSetFromTimePoints();

	public abstract DataSet<SparseInstance> createDataSetFromTimePoints();

	public abstract void pushUp();

	public abstract void deleteSequence();

	public abstract void levelTimeWindow();

	public abstract void levelCenterOfMass(Features features);

	public abstract void levelStd();

	public abstract void levelMin();

	public abstract void levelMax();

	public abstract void levelAverage();

	public abstract Set<String> features();

	public abstract void doubleSequence();

	public abstract void sortTimePoints(final String feature);

	public abstract void addGroundTruth(int t, int f, double result);

	public abstract void initGroundTruthSequence();

	public abstract double groundTruthAt(int t, int f);

	public abstract double get(int t, String feature);

	public abstract void put(int t, String feature, double value);

	public abstract int getSequenceLength();

	public abstract void addTimePoint(Map<String, Double> point);

}