package com.naimur978.heartfailurepredictionml

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.naimur978.heartfailurepredictionml.ml.HeartDisease
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var button : Button = findViewById<Button>(R.id.button)
        button.setOnClickListener(View.OnClickListener {

            var ed1 : EditText = findViewById(R.id.editTextNumberDecimal)
            var ed2 : EditText = findViewById(R.id.editTextNumberDecimal2)
            var ed3 : EditText = findViewById(R.id.editTextNumberDecimal3)
            var ed4 : EditText = findViewById(R.id.editTextNumberDecimal4)

            var v1: Float = ed1.text.toString().toFloat()
            var v2: Float = ed2.text.toString().toFloat()
            var v3: Float = ed3.text.toString().toFloat()
            var v4: Float = ed4.text.toString().toFloat()



            var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(4*4)
            byteBuffer.putFloat(v1)
            byteBuffer.putFloat(v2)
            byteBuffer.putFloat(v3)
            byteBuffer.putFloat(v4)



            val model = HeartDisease.newInstance(this)


            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var tv : TextView = findViewById(R.id.textView)

            tv.setText("No : " + outputFeature0[0].toString() + "\n" +
                    "Yes : " + outputFeature0[1].toString() + "\n"
            )

            model.close()
        })

    }
}


