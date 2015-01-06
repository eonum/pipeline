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
	 * Set the value of feature f at time point t. The sequence length must be
	 * at least t+1. The existing value will be overwritten.
	 * 
	 * @param t
	 * @param feature
	 * @param value
	 */
	public abstract void put(int t, String feature, double value);
	
	/**
	 * Get the input value for feature f at time point t.
	 * @param t
	 * @param f
	 * @return
	 */
	public abstract double get(int t, String feature);

	/** get the input sequence length. */
	public abstract int getSequenceLength();

	/** append a time point to the end of the input sequence. */
	public abstract void addTimePoint(Map<String, Double> point);

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
	 * Set the value of ground truth feature f at time point t. The ground truth
	 * sequence length must be at least t+1. The existing value will be
	 * overwritten.
	 * 
	 * @param t
	 * @param f
	 * @param value
	 */
	public abstract void addGroundTruth(int t, int f, double result);

	/**
	 * Initialize an empty ground truth sequence (all values are NaN) with the
	 * same dimensions as the input sequence.
	 */
	public abstract void initGroundTruthSequence();

	/**
	 * Get the ground truth value for feature f at time point t.
	 * @param t
	 * @param f
	 * @return
	 */
	public abstract double groundTruthAt(int t, int f);

	/**
	 * Get the number of dimensions in the output/ground truth sequence. Returns
	 * 1 if there is no ground truth sequence.
	 * 
	 * @return
	 */
	public abstract int outputSize();

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
	 * Get the result value for feature f at time point t.
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

	/** Deep copy. */
	public abstract Sequence copy();
	
	/** delete all input sequence data. */
	public abstract void clearSequence();

	/**
	 * group / add all time points with the same value of feature groupFeature.
	 * After grouping, the sequence is sorted in ascending order according
	 * groupFeature.
	 * 
	 * @param groupFeature
	 */
	public abstract void groupBy(String groupFeature);

	/** Mix all time points. Mix the temporal order. */
	public abstract void shuffle();

	/**
	 * Put all information in the instance (information about the sequence as a
	 * whole) at the bottom/start of the time series.
	 */
	public abstract void putMasterAtBottom();

	/**
	 * Put all information in the instance (information about the sequence as a
	 * whole) on top/at the end of the time series.
	 */
	public abstract void pushMasterOnTop();
	
	/**
	 * Add all information in the instance (information about the sequence as a
	 * whole) to every point in the time series.
	 */
	public abstract void pushUp();

	/**
	 * Invert the temporal order of the input sequence. The ground truth
	 * sequence is not changed!
	 */
	public abstract void invertTemporalOrder();

	/**
	 * Get a data set where each time point is a single instance. The instances
	 * hold the same feature maps as the time points.
	 * 
	 * You can use this to apply basic feature transformers. (Normalization, PCA
	 * ..)
	 * 
	 * @return
	 */
	public abstract DataSet<SparseInstance> getDataSetFromTimePoints();

	/**
	 * Get a data set where each time point is a single instance. All data is
	 * copied and no references to this sequence is kept.
	 * 
	 * @return
	 */
	public abstract DataSet<SparseInstance> createDataSetFromTimePoints();

	/** delete the input sequence. */
	public abstract void deleteSequence();

	/**
	 * Put all information in the input sequence into the instance by adding a
	 * prefix t to each feature. The prefix is equals to each time point.
	 */
	public abstract void levelTimeWindow();

	/**
	 * Get the normed (by sequence length) center of mass of all features in the
	 * given features object.
	 * 
	 * @param features
	 */
	public abstract void levelCenterOfMass(Features features);

	/**
	 * Get the standard deviation of each feature in the sequence and store it
	 * in the instance (master data).
	 */
	public abstract void levelStd();

	/**
	 * Get the minimum of each feature in the sequence and store it in the
	 * instance (master data).
	 */
	public abstract void levelMin();

	/**
	 * Get the maximum of each feature in the sequence and store it in the
	 * instance (master data).
	 */
	public abstract void levelMax();

	/**
	 * Get the average of each feature in the sequence and store it in the
	 * instance (master data).
	 */
	public abstract void levelAverage();

	@Override
	public abstract Set<String> features();

	/**
	 * Duplicate the input sequence. Append a copy to the end.
	 */
	public abstract void doubleSequence();
	
	/**
	 * Sort the time points according the specified feature in ascending order.
	 * @param feature
	 */
	public abstract void sortTimePoints(final String feature);
	
	/**
	 * Create a ground truth sequence for standard prediction tasks where the
	 * outcome should be identical to the future input. E.g. Weather forecast
	 * based on current weather.
	 * 
	 * @param timeLag number of time points we want to look into the future.
	 * @param features features that should be predicted. (>= 1)
	 */
	public abstract void createTargetForPrediction(int timeLag, Features features);

}