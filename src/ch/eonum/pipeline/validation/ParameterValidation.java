package ch.eonum.pipeline.validation;

import java.util.Map;

import ch.eonum.pipeline.core.Parameters;

/**
 * Class holding all information about a certain parameter which is to be validated.
 * This includes the range (min - max), the starting range for gradient ascent/descent (start, end),
 * the initial value of the parameter, the step size and a flag indicating all values to be
 * in log2 space or not.
 * 
 * @author tim
 *
 */
public class ParameterValidation {

	private Parameters[] parameters;
	private String name;
	private double start;
	private double end;
	private double min;
	private double max;
	private double initialvalue;
	private double stepSize;
	private boolean log2Space;
	private double range;

	/**
	 * 
	 * @param parameters all parameter objects which are to be updated
	 * @param name the parameter's name
	 * @param start start range
	 * @param end end range
	 * @param min minimum value
	 * @param max maximum value
	 * @param initialValue initial value
	 * @param stepSize step size
	 * @param log2Space are all the above values in log2 space or not
	 */
	public ParameterValidation(Parameters[] parameters, String name,
			double start, double end, double min, double max, double initialValue, double stepSize, boolean log2Space) {
		this.parameters = parameters;
		this.name = name;
		this.start = start;
		this.end = end;
		this.min = min;
		this.max = max;
		this.initialvalue = initialValue;
		this.stepSize = stepSize;
		this.log2Space = log2Space;
		this.range = end - start;
	}
	
	public Parameters[] getParameters() {
		return parameters;
	}

	public void setParameters(Parameters[] parameters) {
		this.parameters = parameters;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public double getStart() {
		return start;
	}

	public void setStart(double start) {
		this.start = start;
	}

	public double getEnd() {
		return end;
	}

	public void setEnd(double end) {
		this.end = end;
	}

	public double getMin() {
		return min;
	}

	public void setMin(double min) {
		this.min = min;
	}

	public double getMax() {
		return max;
	}

	public void setMax(double max) {
		this.max = max;
	}

	public double getInitialvalue() {
		return initialvalue;
	}

	public void setInitialvalue(double initialvalue) {
		this.initialvalue = initialvalue;
	}

	public double getStepSize() {
		return stepSize;
	}

	public void setStepSize(double stepSize) {
		this.stepSize = stepSize;
	}

	public boolean isLog2Space() {
		return log2Space;
	}

	public void setLog2Space(boolean log2Space) {
		this.log2Space = log2Space;
	}

	public double getRange() {
		return this.range;
	}

	/**
	 * set all parameters to the optimised value;
	 * @param params
	 */
	public static void updateParameters(Map<ParameterValidation, Double> params) {
		for(ParameterValidation each : params.keySet())
			for(Parameters p : each.getParameters())
				if(each.isLog2Space())
					p.putParameter(each.getName(), Math.pow(2, params.get(each)));
				else
					p.putParameter(each.getName(), params.get(each));
	}

}
