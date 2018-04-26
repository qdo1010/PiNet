//
//  PopUpView.swift
//  Koloda_Example
//
//  Created by Quan Do on 4/22/18.
//  Copyright © 2018 CocoaPods. All rights reserved.
//

import UIKit

class PopUpView: UIViewController {
    var gameTimer: Timer! //Timer object

    @IBOutlet var l1: UILabel!
    @IBOutlet var l2: UILabel!
    @IBOutlet var l3: UILabel!
    @IBOutlet var l4: UILabel!
    var counter = 0;
    override func viewDidLoad() {
        super.viewDidLoad()
        gameTimer = Timer.scheduledTimer(timeInterval: 2, target: self, selector: #selector(timeaction), userInfo: nil, repeats: true)
        l1.isHidden = false
        l2.isHidden = true
        l3.isHidden = true
        l4.isHidden = true

        
        // Do any additional setup after loading the view.
    }
    //Timer action
    @objc func timeaction(){
        counter = counter + 1;
        //code for move next VC
        if (counter == 1){
            l1.isHidden = true
            l2.isHidden = false
            l3.isHidden = true
            l4.isHidden = true
        }
        else if(counter == 2){
            l1.isHidden = true
            l2.isHidden = true
            l3.isHidden = false
            l4.isHidden = true
        }
        else if(counter == 3){
            l1.isHidden = true
            l2.isHidden = true
            l3.isHidden = true
            l4.isHidden = false
        }
        else if (counter >= 4){
        self.performSegue(withIdentifier: "SecondViewController", sender: self)
            counter = 0;
            gameTimer.invalidate()//after that timer invalid

        }
        //let secondVC = storyboard?.instantiateViewController(withIdentifier: "SecondViewController") as! BackgroundAnimationViewController
        //self.navigationController?.pushViewController(secondVC, animated: true)
        
    }
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
