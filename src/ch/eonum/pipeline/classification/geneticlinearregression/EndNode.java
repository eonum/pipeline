package ch.eonum.pipeline.classification.geneticlinearregression;

/**
 * Feature end node. Returns the value of a feature.
 * @author tim
 *
 */
public class EndNode extends Node {

	private String feature;
	private int featureIndex;

	public EndNode(String feature, int featureIndex) {
		this.feature = feature;
		this.featureIndex = featureIndex;
	}
	
	@Override
	public String toString(){
		return feature;
	}

	@Override
	public int getDepth() {
		return 1;
	}

	@Override
	public double evaluate(double[] data) {
		return data[featureIndex];
	}

	@Override
	public Node copy() {
		return new EndNode(feature, featureIndex);
	}

	@Override
	public String getNodeName() {
		return "endNode";
	}

}
