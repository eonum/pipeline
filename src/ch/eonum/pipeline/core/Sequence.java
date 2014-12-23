package ch.eonum.pipeline.core;

import java.util.Map;
import java.util.Set;

public abstract class Sequence extends SparseInstance {

	public Sequence(String id, String gt, Map<String, Double> vector) {
		super(id, gt, vector);
	}

	public Sequence(String id, String groundTruth, Map<String, Double> vector,
			String className) {
		super(id, groundTruth, vector, className);
	}

	public abstract int getGroundTruthLength();

	public abstract boolean hasGroundTruthSequence();

	public abstract int outputSize();

	public abstract void createTargetForPrediction(int timeLag, Features features);

	public abstract boolean hasSequenceTarget();

	public abstract double[][] getDenseRepresentation(Features features);

	public abstract double resultAt(int t, int f);

	public abstract void addSequenceResult(int t, int f,
			double result);

	public abstract void initSequenceResults();

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