//
//  BackgroundAnimationViewController.swift
//  Koloda
//
//  Created by Eugene Andreyev on 7/11/15.
//  Copyright (c) 2015 CocoaPods. All rights reserved.
//

import UIKit
import Koloda
import pop
import CoreML

private let numberOfCards: Int = 5
private let frameAnimationSpringBounciness: CGFloat = 9
private let frameAnimationSpringSpeed: CGFloat = 16
private let kolodaCountOfVisibleCards = 2
private let kolodaAlphaValueSemiTransparent: CGFloat = 0.1

@available(iOS 11.0, *)
class BackgroundAnimationViewController: UIViewController {
    //for detection
    var inputImage: CGImage!
    var counter = 0
    var openDrawing = 0
    let model = mnistCNN()
    var choice: [Character] = []
    @IBOutlet var score: UILabel!
    @IBOutlet var end: UILabel!
    @IBOutlet var drawView: DrawView!
    //  @IBOutlet var drawView: UIView!
    @IBOutlet weak var kolodaView: CustomKolodaView!
    
    @IBOutlet var predictLabel: UILabel!
    //MARK: Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        kolodaView.alphaValueSemiTransparent = kolodaAlphaValueSemiTransparent
        kolodaView.countOfVisibleCards = kolodaCountOfVisibleCards
        kolodaView.delegate = self
        kolodaView.dataSource = self
        kolodaView.animator = BackgroundKolodaAnimator(koloda: kolodaView)
        kolodaView.isHidden = false
        drawView.isHidden = false
        end.isHidden = true
        predictLabel.isHidden = true
        score.isHidden = true
        drawView.autoresizingMask = [.flexibleWidth, .flexibleBottomMargin]

        openDrawing = 0
        self.modalTransitionStyle = UIModalTransitionStyle.flipHorizontal
    }
    
    
    //MARK: IBActions
    @IBAction func leftButtonTapped() {
        kolodaView?.swipe(.left)
        drawView.isHidden = true
        predictLabel.isHidden = true

        drawView.lines = []
        counter = 0
    }
    
    @IBAction func rightButtonTapped() {
        let context = drawView.getViewContext()
        inputImage = context?.makeImage()
        let pixelBuffer = UIImage(cgImage: inputImage).pixelBuffer()
        let output = try? model.prediction(image: pixelBuffer!)
        score.text = String(choice) + (output?.classLabel)!
        predictLabel.text = (output?.classLabel)!
        if (counter == 1){
            choice.append(Character("."))
        }
        choice.append(Character((output?.classLabel)!))
        if (counter != 0){
            if (Int((output?.classLabel)!)! == piToN(to: counter)){
                print("cool")
                drawView.lines = []
                drawView.setNeedsDisplay()
                predictLabel.isHidden = true
                
            }
            else {
                print("fail")
                kolodaView.isHidden = true
                drawView.isHidden = true
                counter = 0
                end.isHidden = false
                end.text = "Nope, this isn't Pi!"
                score.isHidden = false


            }
        }
        else {
            if (Int((output?.classLabel)!)! == 3){
                print("cool")
                drawView.lines = []
                drawView.setNeedsDisplay()
                predictLabel.isHidden = true
            }
            else{
                print("fail")
                counter = 0
                kolodaView.isHidden = true
                drawView.isHidden = true

                end.isHidden = false
                end.text = "Nope, this isn't Pi!"
                score.isHidden = false

            }
        }
        drawView.lines = []
        predictLabel.isHidden = false
        counter = counter + 1
        kolodaView?.swipe(.right,force: true)
        drawView.isHidden = true
        
    }
    
    @IBAction func undoButtonTapped() {
        kolodaView?.revertAction()
    }
    
    func piToN(to n: Int) -> Int {
        //let factor = (pow(10, n) as NSDecimalNumber).doubleValue
        let pi = String(Double.pi)
        var stringResult: String = ""
        stringResult = String(pi.prefix(n+2))
        let lastChar = stringResult.last!   // lastChar
        if(n < 1 || n > 14){
            print("Please input range from 1 to 14")
        }else{
            print(stringResult)
        }
        return Int(String(lastChar))!
    }
    
}

//MARK: KolodaViewDelegate
extension BackgroundAnimationViewController: KolodaViewDelegate {
    
    func kolodaDidRunOutOfCards(_ koloda: KolodaView) {
        kolodaView.resetCurrentCardIndex()
    }
    func koloda(_ koloda: KolodaView, shouldSwipeCardAt index: Int, in direction: SwipeResultDirection) -> Bool{
        if (openDrawing  == 0){
            let alert = UIAlertController(title: "Sorry", message: "You need to write a digit in Pi first!", preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "Sure", style: UIAlertActionStyle.default, handler: nil))
            self.present(alert, animated: true, completion: nil)
            return false
        }
        else {
            return true
        }
    }


    func koloda(_ koloda: KolodaView, didSelectCardAt index: Int) {
        openDrawing = 1
       // UIApplication.shared.openURL(URL(string: "https://yalantis.com/")!)
        print("hi")
        //koloda.setNeedsDisplay()
        drawView.isHidden = false
        koloda.bringSubview(toFront: drawView)
        drawView.autoresizingMask = [.flexibleWidth, .flexibleBottomMargin]
        drawView.becomeFirstResponder()
        //drawView.isHidden = false
        //self.view.bringSubview(toFront: drawView)
      //  kolodaView.isHidden = true


    }
    
    func kolodaShouldApplyAppearAnimation(_ koloda: KolodaView) -> Bool {
        return true
    }
    
    func kolodaShouldMoveBackgroundCard(_ koloda: KolodaView) -> Bool {
        return false
    }
    
    func kolodaShouldTransparentizeNextCard(_ koloda: KolodaView) -> Bool {
        return true
    }
    
    func koloda(kolodaBackgroundCardAnimation koloda: KolodaView) -> POPPropertyAnimation? {
        let animation = POPSpringAnimation(propertyNamed: kPOPViewFrame)
        animation?.springBounciness = frameAnimationSpringBounciness
        animation?.springSpeed = frameAnimationSpringSpeed
        return animation
    }
}

// MARK: KolodaViewDataSource
extension BackgroundAnimationViewController: KolodaViewDataSource {
    
    func kolodaSpeedThatCardShouldDrag(_ koloda: KolodaView) -> DragSpeed {
        return .default
    }
    
    func kolodaNumberOfCards(_ koloda: KolodaView) -> Int {
        openDrawing = 0
        return numberOfCards
    }
    
    func koloda(_ koloda: KolodaView, viewForCardAt index: Int) -> UIView {
        openDrawing = 0
        return UIImageView(image: UIImage(named: "cards_\(index + 1)"))
    }
    
    func koloda(_ koloda: KolodaView, viewForCardOverlayAt index: Int) -> OverlayView? {
        openDrawing = 0
        return Bundle.main.loadNibNamed("CustomOverlayView", owner: self, options: nil)?[0] as? OverlayView
    }
}
extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_OneComponent8,
                                         attrs,
                                         &pixelBuffer)
        
        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)
        
        let grayColorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: grayColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                                        return nil
        }
        
        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
}
