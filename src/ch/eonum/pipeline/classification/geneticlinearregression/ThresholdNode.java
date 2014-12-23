package ch.eonum.pipeline.classification.geneticlinearregression;

public class ThresholdNode extends Node {

	private String feature;
	private int featureIndex;
	/** threshold range: [0,1] */
	private double threshold;

	public ThresholdNode(String feature, int featureIndex, double threshold) {
		this.feature = feature;
		this.featureIndex = featureIndex;
		this.threshold = threshold;
	}
	
	@Override
	public String toString(){
		return feature + " > " + threshold;
	}

	@Override
	public int getDepth() {
		return 1;
	}

	@Override
	public double evaluate(double[] data) {
		return data[featureIndex] > threshold ? 1.0 : 0.0;
	}

	@Override
	public Node copy() {
		return new ThresholdNode(feature, featureIndex, threshold);
	}

	@Override
	public String getNodeName() {
		return "thresholdNode";
	}

}
