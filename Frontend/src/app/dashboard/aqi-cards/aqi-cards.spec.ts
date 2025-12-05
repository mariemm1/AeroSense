import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AqiCards } from './aqi-cards';

describe('AqiCards', () => {
  let component: AqiCards;
  let fixture: ComponentFixture<AqiCards>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AqiCards]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AqiCards);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
