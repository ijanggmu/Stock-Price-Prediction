# from matplotlib import image
# import matplotlib as plt
# import base64
# from io import BytesIO

# def get_graph():
#     buffer= BytesIO()
#     plt.savefig(buffer,format='png')
#     buffer.seek(0)
#     image_png=buffer.getvalue()
#     graph=base64.b64decode(image_png)
#     graph=graph.decode('utf-8')
#     buffer.close()
#     return graph

# def get_plot(dtp,predictions):
#     plt.switch_backend('AGG')
#     plt.title('Nabil Stock Price Prediction')
#     plt.xlabel('Date')
#     plt.ylabel('Nabil Stock Price')
#     plt.plot(dtp, color='blue', label='Actual Nabil Stock Price')
#     plt.plot(predictions, color='red', label='Predicted Nabil Stock Price')
#     plt.legend()
#     plt.figure(figsize=(10, 6))
#     graph=get_graph()
#     return graph