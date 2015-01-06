package ch.eonum.pipeline.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * Wrapper class for the gnuplot command line tool.
 * @author tim
 *
 */
public class Gnuplot {

	private static final String AreaUnderCurveTemplate = "set title '@title'\n" + 
			"set yrange [0:1.0]\n" + 
			"set xrange [0:1.0]\n" + 
			"set pointsize 0.01\n" + 
			"set grid\n" + 
			"set xlabel 'Anteil Plausible in Auswahl'\n" + 
			"set ylabel 'Anteil Unplausible in Auswahl'\n" + 
			"set nokey\n" + 
			"\n" + 
			"set terminal png  \n" + 
			"set output \"@out_file\" \n" + 
			"\n" + 
			"plot '@in_file' with lines";
	private static final String HistogramTemplate = "" + 
			"set term png\n" + 
			"set output '@out_file'\n" + 
			"set key off\n" + 
			"set style fill \n" + 
			"set boxwidth 0.01\n" + 
			"set style histogram clustered gap 0.01 title offset 0.01,0.005\n" + 
			"set autoscale\n" + 
			"\n" + 
			"plot @files"; // '@in_file1' with boxes, '@in_file2' with boxes
	private static final String OneDimensionalCurveTemplate = "set title '@title'\n" + 
			"set autoscale\n" + 
			"set xlabel '@xlabel'\n" + 
			"set pointsize 1.0\n" + 
			"set grid\n" + 
			"set ylabel 'Accuracy'\n" + 
			"set nokey\n" + 
			"set terminal png  \n" + 
			"set output \"@out_file\" \n" + 
			"\n" + 
			"plot '@in_file' with lines" ;
	private static final String GridTemplate = "" +
			"set terminal png \n" + 
			"set output '@out_file'\n" + 
			"set title '@title'\n" + 
			"set border 4095 front linetype -1 linewidth 1.000\n" + 
			"set view map\n" + 
			"set isosamples 100, 100\n" + 
			"unset surface\n" + 
			"set style data pm3d\n" + 
			"set style function pm3d\n" + 
			"set ticslevel 0\n" + 
			"set autoscale\n" + 
			"set pm3d implicit at b\n" + 
			"set palette negative nops_allcF maxcolors 0 gamma 1.5 gray\n" + 
			"plot '@in_file' with image\n" + 
			"";
	
	private static final String ScatterPlotTemplate = 
			"set title \"Scatter Plot\"\n" + 
			"set autoscale\n" + 
			"set terminal png  \n" + 
			"set output \"@out_file\"\n" +  
			"plot '@in_file' with points";
	
	private static final String ScatterPlotLabelsTemplate = 
		"set title \"Scatter Plot\"\n" + 
		"set autoscale\n" + 
		"set terminal png enhanced size 1600,1600\n" + 
		"set output \"@out_file\"\n" +  
		"LabelDRG(Label,Size) = sprintf(\"{/=%d %s}\", Size, Label)\n" +
		"plot '@in_file' using 1:2:(LabelDRG(stringcolumn(3),$4)):(stringcolumn(5)) with labels ";
	
	private static final String HeatMapTemplate = 
		"set title \"Heat Map\"\n" + 
		"unset key\n" + 
		"set dgrid3d\n" + 
		"set hidden3d\n" + 
		"set pm3d map\n" +
		"set autoscale\n" + 
		"set terminal png\n" + 
		"set output \"@out_file\"\n" +  
		"splot '@in_file'";

	public static void plot(String template, String plotFile,
			HashMap<String, String> params) {
		try {
			FileWriter fstream = new FileWriter(plotFile);
			BufferedWriter plot_file = new BufferedWriter(fstream);

			FileInputStream fistream = new FileInputStream(template);
			DataInputStream in = new DataInputStream(fistream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			while ((strLine = br.readLine()) != null) {
				for (String key : params.keySet())
					strLine = strLine.replace(key, params.get(key));
				plot_file.write(strLine + "\n");
			}
			// Close the input stream
			in.close();
			plot_file.close();

			Shell.executeCommand("gnuplot", "gnuplot " + plotFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void plotAreaUnderCurve(String fileName, String title) {
		Gnuplot.plotCurve(fileName, title, Gnuplot.AreaUnderCurveTemplate);
	}
	
	public static void plotOneDimensionalCurve(String fileName, String title) {
		Gnuplot.plotCurve(fileName, title, Gnuplot.OneDimensionalCurveTemplate);
	}

	public static void plotHistogram(String[] fileList, String fileName) {
		try {
			FileWriter fstream = new FileWriter(fileName + ".plt");
			BufferedWriter plotFile = new BufferedWriter(fstream);

			String plotString = Gnuplot.HistogramTemplate.replace("@out_file", fileName + ".png");
			String files = "";
			for(String each : fileList)
				files += "'" + each + "' with boxes, ";
			// remove last comma
			files = files.substring(0, files.length() -2);
			plotString = plotString.replace("@files", files);
			plotFile.write(plotString);
			plotFile.close();

			Shell.executeCommand("gnuplot", "gnuplot " + fileName + ".plt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void plotCurve(String fileName, String title, String template) {
		try {
			FileWriter fstream = new FileWriter(fileName + ".plt");
			BufferedWriter plotFile = new BufferedWriter(fstream);

			String plotString = template.replace(
					"@title", title);
			plotString = plotString.replace("@out_file", fileName + ".png");
			plotString = plotString.replace("@in_file", fileName);
			plotFile.write(plotString);
			plotFile.close();

			Shell.executeCommand("gnuplot", "gnuplot " + fileName + ".plt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void plotGrid(String fileName, String title) {
		try {
			FileWriter fstream = new FileWriter(fileName + ".plt");
			BufferedWriter plotFile = new BufferedWriter(fstream);

			String plotString = Gnuplot.GridTemplate.replace(
					"@title", title);
			plotString = plotString.replace("@out_file", fileName + ".png");
			plotString = plotString.replace("@in_file", fileName);
			plotFile.write(plotString);
			plotFile.close();

			Shell.executeCommand("gnuplot", "gnuplot " + fileName + ".plt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void plotScatterplot(String fileName, String title) {
		Gnuplot.plotCurve(fileName, title, Gnuplot.ScatterPlotTemplate);
	}

	public static void plotScatterplotWithLabels(String fileName, String title) {
		Gnuplot.plotCurve(fileName, title, Gnuplot.ScatterPlotLabelsTemplate);
	}

	public static void plotHeatMap(String fileName, String title) {
		Gnuplot.plotCurve(fileName, title, Gnuplot.HeatMapTemplate);
	}

	public static void plotOneDimensionalCurve(
			Map<Integer, Double> map, String title, String fileName) {
		try {
			FileWriter fstream = new FileWriter(fileName);
			BufferedWriter out = new BufferedWriter(fstream);
			for(Integer each : map.keySet()){
				out.write(each + "	" + map.get(each) + "\n");
			}
			out.close();
		} catch (Exception exc) {
			exc.printStackTrace();
		}
		Gnuplot.plotOneDimensionalCurve(fileName, title);
	}
}
