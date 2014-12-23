package ch.eonum.pipeline.features;


public class FeatureDelta implements Comparable<FeatureDelta> {
	private String feature;
	private double delta;
	public FeatureDelta(String feature, double delta){
		this.feature = feature;
		this.delta = delta;
	}
	
	public String getFeature() {
		return feature;
	}
	
	public double getDelta() {
		return delta;
	}

	@Override
	public int compareTo(FeatureDelta other) {
		return delta - other.getDelta() > 0 ? -1 : 1;
	}
	
	@Override
	public String toString(){
		return this.feature + ": " + this.delta;
	}
}

