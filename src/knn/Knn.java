package knn;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author x0r
 */
public class Knn {

    //how many closest neighbours do we pick
    static final int k = 5;

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("./iris.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        int numAttrs = data.numAttributes();
        Instance test = data.instance(2);
        data.delete(2);
        Item testItem = new Item(test, numAttrs);
        testItem.Label = null;

        int numInstances = data.numInstances();
        ArrayList<Item> Items = new ArrayList<>();

        for (int instIdx = 0; instIdx < numInstances; instIdx++) {
            Items.add(new Item(data.instance(instIdx), numAttrs));
            //Items.get(Items.size()-1).showData();
        }

        KnnClassifier knn = new KnnClassifier(Items, k);
        int found = knn.Evaluate(testItem);
        System.out.println("Result with my classifier: " + found);

        Classifier ibk = new IBk();
        ibk.buildClassifier(data);

        double class1 = ibk.classifyInstance(test);
        System.out.println("Result with ibk from weka: " + (int) class1);

    }

}
