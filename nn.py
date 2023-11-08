import larq as lq
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model


    
def nn(bitwidth, LUT,class_number):
    

    if LUT == 1:
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_1')(model1_in_1)

        model1_in_2 = Input(shape=(1,),name='input2')
        model1_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_2')(model1_in_2)
        concatenated_1 = Concatenate()([model1_in_1_quant, model1_in_2_quant])

        model1_out_1 = Dense(64, activation='relu', name='layer_1_1')(concatenated_1)
        bn1 = BatchNormalization()(model1_out_1)
        model1_out_11 = Dense(32, activation='relu', name='output_layer_1_11')(bn1)
        bn2 = BatchNormalization()(model1_out_11)
        model1_out_2 = Dense(class_number, activation='softmax', name='output_layer_1_2')(bn2)
        model1_out_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_output_1')(model1_out_2)
        #model1_out_11 = Dense(32, activation='relu', name='layer_1_11')(model1_out_1)
        #model1_out_2_quant = lq.layers.QuantDense(1, activation='relu', name='layer_1_2', **kwargs)(model1_out_1)
        model = Model([model1_in_1,model1_in_2], model1_out_2_quant)
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        return Model([model1_in_1,model1_in_2], model1_out_2)
    
    if LUT == 5:

        # MODEL 1
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_11')(model1_in_1)
        model1_in_2 = Input(shape=(1,),name='input2')
        model1_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_12')(model1_in_2)
        concatenated_1 = Concatenate()([model1_in_1_quant, model1_in_2_quant])


        #model1_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_1_1', **kwargs)(concatenated_1)
        model1_out_1 = Dense(64, activation='relu', name='layer_1_1')(concatenated_1)
        bn = BatchNormalization(scale=False,center=False)(model1_out_1)
        model1_out_11 = Dense(32, activation='relu', name='layer_1_11')(bn)
        bn2 = BatchNormalization(scale=False,center=False)(model1_out_11)
        model1_out_2 = Dense(1, activation='relu', name='layer_1_2')(bn2)

        model1 = Model([model1_in_1,model1_in_2], model1_out_2)

        
        
        # MODEL 2
        model2_in_1 = Input(shape=(1,),name='input3')
        model2_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_21')(model2_in_1)
        model2_in_2 = Input(shape=(1,),name='input4')
        model2_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_22')(model2_in_2)
        concatenated_2 = Concatenate()([model2_in_1_quant, model2_in_2_quant])


        #model2_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_2_1', **kwargs)(concatenated_2)
        model2_out_1 = Dense(64, activation='relu', name='layer_2_1')(concatenated_2)
        bn = BatchNormalization(scale=False,center=False)(model2_out_1)
        model2_out_11 = Dense(32, activation='relu', name='layer_2_11')(bn)
        bn2 = BatchNormalization(scale=False,center=False)(model2_out_11)
        model2_out_2 = Dense(1, activation='relu', name='layer_2_2')(bn2)

        model2 = Model([model2_in_1,model2_in_2], model2_out_2)

        
        
        # MODEL 3
        model3_in_1 = Input(shape=(1,),name='input5')
        model3_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_31')(model3_in_1)
        model3_in_2 = Input(shape=(1,),name='input6')
        model3_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_32')(model3_in_2)
        concatenated_3 = Concatenate()([model3_in_1_quant, model3_in_2_quant])

        #model3_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_3_1', **kwargs)(concatenated_3)
        model3_out_1 = Dense(64, activation='relu', name='layer_3_1')(concatenated_3)
        bn = BatchNormalization(scale=False,center=False)(model3_out_1)
        model3_out_11 = Dense(32, activation='relu', name='layer_3_11')(bn)
        bn2 = BatchNormalization(scale=False,center=False)(model3_out_11)
        model3_out_2 = Dense(1, activation='relu', name='layer_3_2')(bn2)

        model3 = Model([model3_in_1,model3_in_2], model3_out_2)

        concatenated_5_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_5_1')(model1.output)
        concatenated_5_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_5_2')(model2.output)
        concatenated_5 = Concatenate()([concatenated_5_quant_0, concatenated_5_quant_1])
        

        model5_out_1  = Dense(64, activation='relu', name='layer_5_1')(concatenated_5)
        bn = BatchNormalization(scale=False,center=False)(model5_out_1)
        model5_out_11 = Dense(32, activation='relu', name='layer_5_11')(bn)
        bn2 = BatchNormalization(scale=False,center=False)(model5_out_11)
        model5_out_2 = Dense(1, activation='relu', name='layer_5_2')(bn2)


        model5 = Model([model1.input, model2.input], model5_out_2)

        concatenated_7_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_7_1')(model3.output)
        concatenated_7_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_7_2')(model5.output)

        concatenated_7 = Concatenate()([concatenated_7_quant_0,concatenated_7_quant_1])

        model7_out_1 =  Dense(128, activation='relu', name='output_layer_7_1',)(concatenated_7)
        bn = BatchNormalization(scale=False,center=False)(model7_out_1)

        model7_out_11 = Dense(64, activation='relu', name='output_layer_7_11')(bn)
        bn1 = BatchNormalization(scale=False,center=False)(model7_out_11)

        model7_out_2 = Dense(class_number, activation='softmax', name='output_layer_7_2')(bn1)
        model7_out_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_output_1')(model7_out_2)

        model7 = Model([model3.input, model5.input], model7_out_2_quant)

        plot_model(model7, to_file='model.png', show_shapes=True, show_layer_names=True)

        return model7

    if LUT == 9:


        # MODEL 1
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_11')(model1_in_1)
        model1_in_2 = Input(shape=(1,),name='input2')
        model1_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_12')(model1_in_2)
        concatenated_1 = Concatenate()([model1_in_1_quant, model1_in_2_quant])


        #model1_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_1_1', **kwargs)(concatenated_1)
        model1_out_1 = Dense(256, activation='relu', name='layer_1_1')(concatenated_1)
        bn_1_1 = BatchNormalization()(model1_out_1)
        model1_out_2 = Dense(128, activation='relu', name='layer_1_12')(bn_1_1)
        bn_1_12 = BatchNormalization()(model1_out_2)
        model1_out_11 = Dense(64, activation='relu', name='layer_1_11')(bn_1_12)
        bn_1_2 = BatchNormalization()(model1_out_11)
        model1_out_2 = Dense(1, activation='relu', name='layer_1_2')(bn_1_2)

        model1 = Model([model1_in_1,model1_in_2], model1_out_2)

        
        
        # MODEL 2
        model2_in_1 = Input(shape=(1,),name='input3')
        model2_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_21')(model2_in_1)
        model2_in_2 = Input(shape=(1,),name='input4')
        model2_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_22')(model2_in_2)
        concatenated_2 = Concatenate()([model2_in_1_quant, model2_in_2_quant])

        #model2_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_2_1', **kwargs)(concatenated_2)
        model2_out_1 = Dense(256, activation='relu', name='layer_2_1')(concatenated_2)
        bn_2_1 = BatchNormalization()(model2_out_1)
        model2_out_12 = Dense(128, activation='relu', name='layer_2_21')(bn_2_1)
        bn_2_12 = BatchNormalization()(model2_out_12)
        model2_out_11 = Dense(64, activation='relu', name='layer_2_11')(bn_2_12)
        bn_2_2 = BatchNormalization()(model2_out_11)
        model2_out_2 = Dense(1, activation='relu', name='layer_2_2')(bn_2_2)

        model2 = Model([model2_in_1,model2_in_2], model2_out_2)

        
        
        # MODEL 3
        model3_in_1 = Input(shape=(1,),name='input5')
        model3_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_31')(model3_in_1)
        model3_in_2 = Input(shape=(1,),name='input6')
        model3_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_32')(model3_in_2)
        concatenated_3 = Concatenate()([model3_in_1_quant, model3_in_2_quant])

        #model3_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_3_1', **kwargs)(concatenated_3)
        model3_out_1 = Dense(256, activation='relu', name='layer_3_1')(concatenated_3)
        bn_3_1 = BatchNormalization()(model3_out_1)
        
        model3_out_12 = Dense(128, activation='relu', name='layer_3_12')(bn_3_1)
        bn_3_12 = BatchNormalization()(model3_out_12)


        model3_out_11 = Dense(64, activation='relu', name='layer_3_11')(bn_3_12)
        bn_3_2 = BatchNormalization()(model3_out_11)
        
        model3_out_2 = Dense(1, activation='relu', name='layer_3_2')(bn_3_2)

        model3 = Model([model3_in_1,model3_in_2], model3_out_2)

        
        
        # MODEL 4
        model4_in_1 = Input(shape=(1,),name='input7')
        model4_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_41')(model4_in_1)
        model4_in_2 = Input(shape=(1,),name='input8')
        model4_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_42')(model4_in_2)
        concatenated_4 = Concatenate()([model4_in_1_quant, model4_in_2_quant])

        #model4_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_4_1', **kwargs)(concatenated_4)
        model4_out_1 = Dense(256, activation='relu', name='layer_4_1')(concatenated_4)
        bn_4_1 = BatchNormalization()(model4_out_1)

        model4_out_12 = Dense(128, activation='relu', name='layer_4_12')(bn_4_1)
        bn_4_12 = BatchNormalization()(model4_out_12)

        model4_out_11 = Dense(64, activation='relu', name='layer_4_11')(bn_4_12)
        bn_4_2 = BatchNormalization()(model4_out_11)
        model4_out_2 = Dense(1, activation='relu', name='layer_4_2')(bn_4_2)

        model4 = Model([model4_in_1,model4_in_2], model4_out_2)


        
        new_model4_in_1 = Input(shape=(1,),name='input9')
        new_model4_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='new_quant_input_41')(new_model4_in_1)
        new_model4_in_2 = Input(shape=(1,),name='input10')
        new_model4_in_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='new-quant_input_42')(new_model4_in_2)
        new_concatenated_4 = Concatenate()([new_model4_in_1_quant, new_model4_in_2_quant])

        #model4_out_1 = lq.layers.QuantDense(256, activation='relu', name='layer_4_1', **kwargs)(concatenated_4)
        new_model4_out_1 = Dense(256, activation='relu', name='new_layer_5_1')(new_concatenated_4)
        new_bn_4_1 = BatchNormalization()(new_model4_out_1)

        model5_out_12 = Dense(128, activation='relu', name='layer_5_12')(new_bn_4_1)
        bn_5_12 = BatchNormalization()(model5_out_12)


        new_model4_out_11 = Dense(64, activation='relu', name='new_layer_4_11')(bn_5_12)
        new_bn_4_2 = BatchNormalization()(new_model4_out_11)
        new_model4_out_2 = Dense(1, activation='relu', name='new_layer_4_2')(new_bn_4_2)

        new_model4 = Model([new_model4_in_1,new_model4_in_2], new_model4_out_2)


        
        # SECOND LAYER MODELS
        # MODEL 5

        concatenated_5_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_5_1')(model1.output)
        concatenated_5_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_5_2')(model2.output)
        concatenated_5 = Concatenate()([concatenated_5_quant_0, concatenated_5_quant_1])

        model5_out_1  = Dense(256, activation='relu', name='layer_5_1')(concatenated_5)
        bn_5_1 = BatchNormalization()(model5_out_1)


        model6_out_12 = Dense(128, activation='relu', name='layer_6_12')(bn_5_1)
        bn_6_12 = BatchNormalization()(model6_out_12)



        model5_out_11 = Dense(64, activation='relu', name='layer_5_11')(bn_6_12)
        bn_5_2 = BatchNormalization()(model5_out_11)
        model5_out_2 = Dense(1, activation='relu', name='layer_5_2')(bn_5_2)

        model5 = Model([model1.input, model2.input], model5_out_2)


    
        # MODEL 6
        concatenated_6_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_6_1')(model3.output)
        concatenated_6_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_6_2')(model4.output)

        concatenated_6 = Concatenate()([concatenated_6_quant_0, concatenated_6_quant_1])


        model6_out_1  = Dense(256, activation='relu', name='layer_6_1')(concatenated_6)
        bn_6_1 = BatchNormalization()(model6_out_1)

        model7_out_12 = Dense(128, activation='relu', name='layer_7_12')(bn_6_1)
        bn_7_12 = BatchNormalization()(model7_out_12)

        model6_out_11 = Dense(64, activation='relu', name='layer_6_11')(bn_7_12)
        bn_6_2 = BatchNormalization()(model6_out_11)
        model6_out_2 = Dense(1, activation='relu', name='layer_6_2')(bn_6_2)


        model6 = Model([model3.input, model4.input], model6_out_2)


        # THIRD LAYER

        

        # FOURTH LAYER MODELS

        # MODEL 7
        concatenated_7_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_7_1')(model5.output)
        concatenated_7_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='quant_input_7_2')(model6.output)
        concatenated_7 = Concatenate()([concatenated_7_quant_0, concatenated_7_quant_1])

        model7_out_3 = Dense(256, activation='relu', name='output_layer_7_3')(concatenated_7)
        bn_7_1 = BatchNormalization()(model7_out_3)

        
        model8_out_12 = Dense(128, activation='relu', name='layer_8_12')(bn_7_1)
        bn_8_12 = BatchNormalization()(model8_out_12)


        model7_out_4 = Dense(64, activation='relu', name='output_layer_7_4')(bn_8_12)
        bn_7_2 = BatchNormalization()(model7_out_4)

        model7_out_5 = Dense(1, activation='relu', name='output_layer_7_5')(bn_7_2)

        model7 = Model([model5.input, model6.input], model7_out_5)



        new_concatenated_7_quant_0 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='new_quant_input_7_1')(model7.output)
        new_concatenated_7_quant_1 = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='new_quant_input_7_2')(new_model4.output)
        new_concatenated_7 = Concatenate()([new_concatenated_7_quant_0, new_concatenated_7_quant_1])

        new_model7_out_3 = Dense(256, activation='relu', name='new_output_layer_7_3')(new_concatenated_7)
        new_bn_7_1 = BatchNormalization()(new_model7_out_3)

        model9_out_12 = Dense(128, activation='relu', name='layer_9_12')(new_bn_7_1)
        bn_9_12 = BatchNormalization()(model9_out_12)

        new_model7_out_4 = Dense(64, activation='relu', name='new_output_layer_7_4')(bn_9_12)
        new_bn_7_2 = BatchNormalization()(new_model7_out_4)

        new_model7_out_5 = Dense(class_number, activation='softmax', name='new_output_layer_7_5')(new_bn_7_2)
        new_model7_out_2_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='new_quant_output_1')(new_model7_out_5)

        new_model7 = Model([model7.input, new_model4.input], new_model7_out_2_quant)

        
        plot_model(new_model7, to_file='model.png', show_shapes=True, show_layer_names=True)
        return new_model7