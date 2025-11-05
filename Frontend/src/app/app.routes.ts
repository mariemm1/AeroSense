import { Routes } from '@angular/router';
import { HomePage } from './home-page/home-page';
import { Signup } from './signup/signup';
import { Signin } from './signin/signin';
import { Dashboard } from './dashboard/dashboard';
import { ContactComponent } from './contact/contact.component';

export const routes: Routes = [
    
  { path: '', redirectTo: 'homPage', pathMatch: 'full' },
  { path: 'signin', component: Signin },
  { path: 'signup', component: Signup },
  { path: 'homPage', component: HomePage},
  { path: 'dashboard', component: Dashboard},
  { path: 'contact', component: ContactComponent },
  { path: '**', redirectTo: 'home' },
];
